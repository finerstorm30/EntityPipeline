import tarfile
import xml.etree.ElementTree as ET
import json
import gzip

from pathlib import Path
from types import SimpleNamespace

DATA_DIR = Path('Data/PubMed/PubmedDocs/')
seen_pmids = set()
seen_pmcs = set()


def set_seen_ids(pmids, pmcs):
    global seen_pmids, seen_pmcs

    seen_pmids = set(pmids)
    seen_pmcs = set(pmcs)


def get_seen_ids():
    return seen_pmids, seen_pmcs
    
def get_pubmed_archives():
    return list(DATA_DIR.rglob("*.tar.gz")) + list(DATA_DIR.rglob("*.xml.gz"))


def get_pubmed_papers(path, paper_batch_size=1000):
    if path.name.endswith(".tar.gz"):
        return get_fulltext_papers(path, paper_batch_size)

    if path.name.endswith(".xml.gz"):
        return get_abstract_papers(path, paper_batch_size)

    raise ValueError(f"Unsupported PubMed file type: {path}")


def get_fulltext_papers(path, paper_batch_size):
    papers = []

    local_pmids = set()
    local_pmcs = set()
    
    with tarfile.open(path, "r:gz") as tar:
        for member in tar:
            with tar.extractfile(member) as fp:
                paper = json.load(fp)

            PMC, PMID = get_pubmed_id(paper)

            already_seen = (
                (PMID and PMID in seen_pmids)
                or (PMC and PMC in seen_pmcs)
                or (PMID and PMID in local_pmids)
                or (PMC and PMC in local_pmcs)
            )

            if already_seen:
                continue
                
            papers.append(
                SimpleNamespace(
                    text=extract_text(paper),
                    pmid=PMID,
                    pmc=PMC,
                )
            )

            if PMID:
                local_pmids.add(PMID)
            if PMC:
                local_pmcs.add(PMC)
    return papers


def get_pubmed_id(paper):
    ids = paper["documents"][0]["passages"][0]["infons"]
    PMC = ids.get("article-id_pmc")
    PMID = ids.get("article-id_pmid")
    return PMC, PMID


def extract_text(paper):    
    passages = paper["documents"][0]["passages"]
    text = ""
    for passage in passages:
        section_type = passage["infons"]["section_type"]
        if section_type == "TABLE" or section_type == "REF":
            continue
        text += passage["text"] + "\n"

    return text


def get_abstract_papers(path, paper_batch_size):
    papers = []

    local_pmids = set()
    local_pmcs = set()

    with gzip.open(path, "rt", encoding="utf-8") as fp:
        tree = ET.parse(fp)

    root = tree.getroot()

    for article in root.findall("PubmedArticle"):
        pmid = article.findtext(".//PMID")

        if (
            (pmid and pmid in seen_pmids)
            or (pmid and pmid in local_pmids)
        ):
            continue

        title = article.findtext(".//ArticleTitle") or ""

        abstract_parts = [
            "".join(abstract_text.itertext())
            for abstract_text in article.findall(".//Abstract/AbstractText")
        ]

        abstract = "\n".join(abstract_parts)

        if not abstract:
            continue

        text = title + "\n" + abstract

        papers.append(
            SimpleNamespace(
                text=text,
                pmid=pmid,
                pmc=None,
            )
        )

        if pmid:
            local_pmids.add(pmid)

    return papers


def mark_papers_seen(papers):
    for paper in papers:
        if paper.pmid:
            seen_pmids.add(paper.pmid)

        if paper.pmc:
            seen_pmcs.add(paper.pmc)


def same_document(current, previous):
    if current.pmid and previous.pmid:
        return current.pmid == previous.pmid

    if current.pmc and previous.pmc:
        return current.pmc == previous.pmc

    return False