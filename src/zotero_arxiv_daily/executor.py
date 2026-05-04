from loguru import logger
from pyzotero import zotero
from omegaconf import DictConfig, ListConfig
from .utils import glob_match
from .retriever import get_retriever_cls
from .protocol import CorpusPaper
import random
from datetime import datetime
from .reranker import get_reranker_cls
from .construct_email import render_email
from .utils import send_email
from openai import OpenAI
from tqdm import tqdm
import json
import re


def normalize_path_patterns(patterns: list[str] | ListConfig | None, config_key: str) -> list[str] | None:
    if patterns is None:
        return None

    if not isinstance(patterns, (list, ListConfig)):
        raise TypeError(
            f"config.zotero.{config_key} must be a list of glob patterns or null, "
            'for example ["2026/survey/**"]. Single strings are not supported.'
        )

    if any(not isinstance(pattern, str) for pattern in patterns):
        raise TypeError(f"config.zotero.{config_key} must contain only glob pattern strings.")

    return list(patterns)


class Executor:
    def __init__(self, config:DictConfig):
        self.config = config
        self.use_zotero = config.executor.get("use_zotero", True)
        self.interest_topics = list(config.executor.get("interest_topics") or [])
        self.interest_keywords = list(config.executor.get("interest_keywords") or [])
        self.require_interest_match = config.executor.get("require_interest_match", False)
        self.llm_relevance_filter = config.executor.get("llm_relevance_filter", False)
        self.llm_relevance_min_score = config.executor.get("llm_relevance_min_score", 7)
        self.llm_relevance_max_candidates = config.executor.get("llm_relevance_max_candidates", 50)
        self.llm_relevance_preview_chars = config.executor.get("llm_relevance_preview_chars", 6000)
        self.include_path_patterns = normalize_path_patterns(config.zotero.include_path, "include_path")
        self.ignore_path_patterns = normalize_path_patterns(config.zotero.ignore_path, "ignore_path")
        self.retrievers = {
            source: get_retriever_cls(source)(config) for source in config.executor.source
        }
        self.reranker = get_reranker_cls(config.executor.reranker)(config) if self.use_zotero else None
        self.openai_client = None
        if config.llm.api.key:
            self.openai_client = OpenAI(api_key=config.llm.api.key, base_url=config.llm.api.base_url)

    def score_paper_by_interest(self, paper) -> float:
        title = (paper.title or "").lower()
        abstract = (paper.abstract or "").lower()
        score = 0.0
        for keyword in self.interest_keywords:
            keyword = keyword.lower().strip()
            if not keyword:
                continue
            score += title.count(keyword) * 3
            score += abstract.count(keyword)
        return score

    def rank_papers_by_interest(self, papers):
        if not self.interest_keywords:
            return papers

        scored_papers = []
        for paper in papers:
            score = self.score_paper_by_interest(paper)
            paper.score = score if score > 0 else None
            scored_papers.append((score, paper))

        matched = [(score, paper) for score, paper in scored_papers if score > 0]
        logger.info(
            f"Interest keyword matching selected {len(matched)} / {len(scored_papers)} papers "
            f"using keywords: {self.interest_keywords}"
        )
        if self.require_interest_match:
            scored_papers = matched

        scored_papers.sort(key=lambda item: item[0], reverse=True)
        return [paper for _, paper in scored_papers]

    def get_paper_relevance_context(self, paper) -> str:
        full_text = paper.full_text or ""
        if len(full_text) > self.llm_relevance_preview_chars:
            full_text = full_text[:self.llm_relevance_preview_chars]
        return (
            f"Title:\n{paper.title}\n\n"
            f"Abstract:\n{paper.abstract}\n\n"
            f"Main content preview:\n{full_text}"
        )

    def judge_paper_relevance_with_llm(self, paper) -> tuple[bool, float, str]:
        topics = "\n".join(f"- {topic}" for topic in self.interest_topics)
        prompt = f"""
You are selecting arXiv papers for a researcher.

Research interests:
{topics}

Decide whether the paper is genuinely relevant to these interests. Prefer papers about WAM / World Action Models, action-conditioned world models, embodied AI, robot learning, vision-language-action models, or LLM agents grounded in perception/action/environments.

Return only strict JSON with this schema:
{{"relevant": true, "score": 0-10, "reason": "one short reason"}}

Paper:
{self.get_paper_relevance_context(paper)}
"""
        response = self.openai_client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a precise research-paper relevance judge. Return only valid JSON.",
                },
                {"role": "user", "content": prompt},
            ],
            **self.config.llm.get("generation_kwargs", {}),
        )
        content = response.choices[0].message.content or ""
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", content, flags=re.DOTALL)
            if match is None:
                raise
            data = json.loads(match.group(0))
        score = float(data.get("score", 0))
        relevant = bool(data.get("relevant", False)) and score >= self.llm_relevance_min_score
        reason = str(data.get("reason", ""))
        return relevant, score, reason

    def filter_papers_by_llm_relevance(self, papers):
        if not self.llm_relevance_filter:
            return papers
        if not self.openai_client:
            logger.warning("LLM relevance filter is enabled, but no LLM API key is configured.")
            return papers
        if not self.interest_topics:
            logger.warning("LLM relevance filter is enabled, but no interest_topics are configured.")
            return papers

        candidates = papers[:self.llm_relevance_max_candidates]
        selected = []
        logger.info(f"Running LLM relevance filter on {len(candidates)} candidate papers.")
        for paper in tqdm(candidates, desc="Filtering papers with LLM"):
            try:
                relevant, score, reason = self.judge_paper_relevance_with_llm(paper)
            except Exception as exc:
                logger.warning(f"Failed to judge relevance for {paper.url}: {exc}")
                continue
            paper.score = score
            if relevant:
                logger.info(f"Selected paper with score {score}: {paper.title} — {reason}")
                selected.append(paper)

        selected.sort(key=lambda paper: paper.score or 0, reverse=True)
        logger.info(
            f"LLM relevance filter selected {len(selected)} / {len(candidates)} papers "
            f"with minimum score {self.llm_relevance_min_score}."
        )
        return selected

    def fetch_zotero_corpus(self) -> list[CorpusPaper]:
        logger.info("Fetching zotero corpus")
        zot = zotero.Zotero(self.config.zotero.user_id, 'user', self.config.zotero.api_key)
        collections = zot.everything(zot.collections())
        collections = {c['key']:c for c in collections}
        corpus = zot.everything(zot.items(itemType='conferencePaper || journalArticle || preprint'))
        corpus = [c for c in corpus if c['data']['abstractNote'] != '']
        def get_collection_path(col_key:str) -> str:
            if p := collections[col_key]['data']['parentCollection']:
                return get_collection_path(p) + '/' + collections[col_key]['data']['name']
            else:
                return collections[col_key]['data']['name']
        for c in corpus:
            paths = [get_collection_path(col) for col in c['data']['collections']]
            c['paths'] = paths
        logger.info(f"Fetched {len(corpus)} zotero papers")
        return [CorpusPaper(
            title=c['data']['title'],
            abstract=c['data']['abstractNote'],
            added_date=datetime.strptime(c['data']['dateAdded'], '%Y-%m-%dT%H:%M:%SZ'),
            paths=c['paths']
        ) for c in corpus]
    
    def filter_corpus(self, corpus:list[CorpusPaper]) -> list[CorpusPaper]:
        if self.include_path_patterns:
            logger.info(f"Selecting zotero papers matching include_path: {self.include_path_patterns}")
            corpus = [
                c for c in corpus
                if any(
                    glob_match(path, pattern)
                    for path in c.paths
                    for pattern in self.include_path_patterns
                )
            ]
        if self.ignore_path_patterns:
            logger.info(f"Excluding zotero papers matching ignore_path: {self.ignore_path_patterns}")
            corpus = [
                c for c in corpus
                if not any(
                    glob_match(path, pattern)
                    for path in c.paths
                    for pattern in self.ignore_path_patterns
                )
            ]
        if self.include_path_patterns or self.ignore_path_patterns:
            samples = random.sample(corpus, min(5, len(corpus)))
            samples = '\n'.join([c.title + ' - ' + '\n'.join(c.paths) for c in samples])
            logger.info(f"Selected {len(corpus)} zotero papers:\n{samples}\n...")
        return corpus

    
    def run(self):
        corpus = []
        if self.use_zotero:
            corpus = self.fetch_zotero_corpus()
            corpus = self.filter_corpus(corpus)
            if len(corpus) == 0:
                logger.error(f"No zotero papers found. Please check your zotero settings:\n{self.config.zotero}")
                return
        else:
            logger.info("Zotero mode is disabled. Papers will be delivered by source/category order.")

        all_papers = []
        for source, retriever in self.retrievers.items():
            logger.info(f"Retrieving {source} papers...")
            papers = retriever.retrieve_papers()
            if len(papers) == 0:
                logger.info(f"No {source} papers found")
                continue
            logger.info(f"Retrieved {len(papers)} {source} papers")
            all_papers.extend(papers)
        logger.info(f"Total {len(all_papers)} papers retrieved from all sources")
        reranked_papers = []
        if len(all_papers) > 0:
            if self.use_zotero:
                logger.info("Reranking papers...")
                reranked_papers = self.reranker.rerank(all_papers, corpus)
            else:
                reranked_papers = self.rank_papers_by_interest(all_papers)
                reranked_papers = self.filter_papers_by_llm_relevance(reranked_papers)
                if len(reranked_papers) == 0 and not self.config.executor.send_empty:
                    logger.info("No papers matched interest filters. No email will be sent.")
                    return
            reranked_papers = reranked_papers[:self.config.executor.max_paper_num]
            if self.openai_client and self.config.executor.get("generate_tldr", True):
                logger.info("Generating TLDR and affiliations...")
                for p in tqdm(reranked_papers):
                    p.generate_tldr(self.openai_client, self.config.llm)
                    if self.config.executor.get("generate_affiliations", True):
                        p.generate_affiliations(self.openai_client, self.config.llm)
            else:
                logger.info("No OpenAI API key provided. Email will use paper abstracts directly.")
                for p in reranked_papers:
                    p.tldr = p.abstract
        elif not self.config.executor.send_empty:
            logger.info("No new papers found. No email will be sent.")
            return
        logger.info("Sending email...")
        email_content = render_email(reranked_papers)
        send_email(self.config, email_content)
        logger.info("Email sent successfully")
