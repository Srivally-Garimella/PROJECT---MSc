"""
Investor Relations Document Collector for TemporalGuard-RAG

Semi-automates collection of official company IR materials such as:
- Annual reports
- Earnings releases
- Investor presentations

This collector is intentionally lightweight and defensive:
- It only uses official company websites
- It stores rich metadata for later review
- It can either just discover links or also download the files

Unlike SEC/XBRL data, IR websites are not standardized across companies, so this
module focuses on "good enough" discovery rather than claiming perfect coverage.
"""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urljoin, urlparse

import requests
import yfinance as yf
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


class IRDocumentCollector:
    """
    Collects official investor relations materials from company websites.

    Best suited for:
    - Discovering official IR pages from company websites
    - Finding annual reports, earnings releases, and presentations
    - Downloading a curated set of documents with metadata
    """

    USER_AGENT = (
        "TemporalGuardRAG/1.0 "
        "(research use; official company investor relations collection)"
    )

    COMMON_IR_PATHS = [
        "/investors",
        "/investor-relations",
        "/investorrelations",
        "/investor",
        "/financials",
        "/annual-report",
        "/annual-reports",
        "/news",
        "/newsroom",
        "/news-releases",
        "/press-releases",
        "/events-presentations",
        "/presentations",
    ]

    CLASSIFICATION_RULES = {
        "annual_report": [
            "annual report",
            "annualreport",
            "form 10-k",
            "10-k",
            "integrated report",
            "shareholder letter",
        ],
        "earnings_release": [
            "earnings release",
            "quarterly results",
            "financial results",
            "results release",
            "earnings results",
            "quarterly report",
        ],
        "investor_presentation": [
            "presentation",
            "investor deck",
            "earnings presentation",
            "investor presentation",
            "slide deck",
        ],
    }

    def __init__(
        self,
        output_dir: str = "data/raw/investor_relations",
        timeout_seconds: int = 15,
        request_delay: float = 0.5,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.timeout_seconds = timeout_seconds
        self.request_delay = request_delay
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": self.USER_AGENT})

        logger.info(f"Initialized IRDocumentCollector with output dir: {self.output_dir}")

    def collect_company_documents(
        self,
        ticker: str,
        website: Optional[str] = None,
        company_name: Optional[str] = None,
        download: bool = False,
        max_documents: int = 30,
    ) -> Dict:
        """
        Discover IR documents for a company and optionally download them.

        Args:
            ticker: Company ticker
            website: Official website (optional; fetched from Yahoo if absent)
            company_name: Company name (optional)
            download: Whether to download discovered files/pages
            max_documents: Maximum number of documents to keep

        Returns:
            Collection summary with discovered documents and local paths
        """
        ticker = ticker.upper().strip()
        profile = self._get_company_profile(ticker, website=website, company_name=company_name)

        if not profile.get("website"):
            return {
                "ticker": ticker,
                "status": "error",
                "error": "Could not determine official company website",
            }

        company_dir = self.output_dir / ticker
        company_dir.mkdir(parents=True, exist_ok=True)

        discovered_pages = self._discover_ir_pages(profile["website"])
        documents = self._discover_documents(
            ticker=ticker,
            company_name=profile.get("company_name"),
            pages=discovered_pages,
            max_documents=max_documents,
        )

        if download:
            downloaded = self._download_documents(ticker, documents)
        else:
            downloaded = []

        summary = {
            "ticker": ticker,
            "company_name": profile.get("company_name"),
            "website": profile.get("website"),
            "status": "success",
            "discovered_pages": discovered_pages,
            "documents_found": len(documents),
            "documents": documents,
            "downloaded_files": downloaded,
            "collected_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }

        manifest_path = company_dir / "manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        logger.info(
            f"IR collection complete for {ticker}: "
            f"{len(documents)} documents discovered, {len(downloaded)} downloaded"
        )
        return summary

    def collect_multiple(
        self,
        tickers: List[str],
        download: bool = False,
        max_documents: int = 30,
        delay_between_companies: float = 1.0,
    ) -> Dict[str, Dict]:
        """Collect IR materials for multiple companies."""
        results: Dict[str, Dict] = {}

        for idx, ticker in enumerate(tickers, start=1):
            logger.info(f"[{idx}/{len(tickers)}] Collecting IR materials for {ticker}")
            try:
                results[ticker] = self.collect_company_documents(
                    ticker=ticker,
                    download=download,
                    max_documents=max_documents,
                )
            except Exception as exc:
                logger.error(f"IR collection failed for {ticker}: {exc}")
                results[ticker] = {
                    "ticker": ticker,
                    "status": "error",
                    "error": str(exc),
                }

            if idx < len(tickers):
                time.sleep(delay_between_companies)

        summary_path = self.output_dir / "collection_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        return results

    def _get_company_profile(
        self,
        ticker: str,
        website: Optional[str] = None,
        company_name: Optional[str] = None,
    ) -> Dict[str, Optional[str]]:
        """Resolve company website and name, preferring local Yahoo cache."""
        if website and company_name:
            return {"website": website, "company_name": company_name}

        cached = self._read_yahoo_cache(ticker)
        if cached:
            info = cached.get("company_info", {})
            return {
                "website": website or info.get("website"),
                "company_name": company_name or info.get("name"),
            }

        try:
            stock = yf.Ticker(ticker)
            info = stock.info or {}
            return {
                "website": website or info.get("website"),
                "company_name": company_name or info.get("longName") or info.get("shortName"),
            }
        except Exception as exc:
            logger.warning(f"Could not fetch company profile for {ticker}: {exc}")
            return {"website": website, "company_name": company_name}

    def _read_yahoo_cache(self, ticker: str) -> Optional[Dict]:
        """Read locally cached Yahoo Finance company profile if present."""
        safe_ticker = ticker.replace(".", "_").replace(":", "_")
        candidate = Path("data/raw/yahoo_finance") / f"{safe_ticker}.json"
        if not candidate.exists():
            return None

        try:
            with open(candidate, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as exc:
            logger.warning(f"Could not read Yahoo cache for {ticker}: {exc}")
            return None

    def _discover_ir_pages(self, website: str) -> List[Dict]:
        """
        Discover likely IR pages from the official company website.

        We probe a small set of common IR paths and keep only pages that load.
        """
        normalized = self._normalize_base_url(website)
        pages: List[Dict] = []
        seen = set()

        for path in [""] + self.COMMON_IR_PATHS:
            url = normalized if not path else urljoin(normalized + "/", path.lstrip("/"))
            if url in seen:
                continue
            seen.add(url)

            response = self._safe_get(url)
            if not response:
                continue

            content_type = response.headers.get("Content-Type", "").lower()
            if "html" not in content_type:
                continue

            title = self._extract_title(response.text)
            pages.append(
                {
                    "url": response.url,
                    "status_code": response.status_code,
                    "title": title,
                }
            )

            time.sleep(self.request_delay)

        return pages

    def _discover_documents(
        self,
        ticker: str,
        company_name: Optional[str],
        pages: List[Dict],
        max_documents: int,
    ) -> List[Dict]:
        """Extract and classify interesting document links from discovered IR pages."""
        documents: List[Dict] = []
        seen_urls = set()

        for page in pages:
            response = self._safe_get(page["url"])
            if not response:
                continue

            soup = BeautifulSoup(response.text, "lxml")
            for link in soup.find_all("a", href=True):
                href = link.get("href", "").strip()
                text = " ".join(link.get_text(" ", strip=True).split())
                absolute_url = urljoin(response.url, href)

                if absolute_url in seen_urls or not self._is_candidate_document_link(absolute_url, text):
                    continue

                seen_urls.add(absolute_url)
                classification, confidence = self._classify_link(text=text, url=absolute_url)
                if not classification:
                    continue

                documents.append(
                    {
                        "ticker": ticker,
                        "company_name": company_name,
                        "source_page": page["url"],
                        "title": text or self._filename_from_url(absolute_url),
                        "url": absolute_url,
                        "document_type": classification,
                        "confidence": confidence,
                        "file_type": self._file_type_from_url(absolute_url),
                    }
                )

                if len(documents) >= max_documents:
                    break

            if len(documents) >= max_documents:
                break

            time.sleep(self.request_delay)

        # Prefer high-confidence documents first.
        documents.sort(
            key=lambda item: (
                0 if item["document_type"] == "annual_report" else 1,
                -item["confidence"],
                item["title"].lower(),
            )
        )
        return documents

    def _download_documents(self, ticker: str, documents: List[Dict]) -> List[Dict]:
        """Download discovered documents/pages into type-specific folders."""
        downloaded: List[Dict] = []
        ticker_dir = self.output_dir / ticker

        for index, doc in enumerate(documents, start=1):
            response = self._safe_get(doc["url"], stream=True)
            if not response:
                continue

            suffix = self._suffix_from_response(doc["url"], response.headers.get("Content-Type", ""))
            target_dir = ticker_dir / doc["document_type"]
            target_dir.mkdir(parents=True, exist_ok=True)
            filename = f"{index:03d}_{self._safe_filename(doc['title'])}{suffix}"
            output_path = target_dir / filename

            try:
                with open(output_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
            except Exception as exc:
                logger.warning(f"Failed to save {doc['url']}: {exc}")
                continue

            downloaded.append(
                {
                    "url": doc["url"],
                    "document_type": doc["document_type"],
                    "path": str(output_path),
                }
            )
            time.sleep(self.request_delay)

        return downloaded

    def _safe_get(self, url: str, stream: bool = False) -> Optional[requests.Response]:
        """Fetch a URL safely and quietly skip problematic pages."""
        try:
            response = self.session.get(
                url,
                timeout=self.timeout_seconds,
                allow_redirects=True,
                stream=stream,
            )
            if response.status_code >= 400:
                return None
            return response
        except requests.RequestException:
            return None

    def _normalize_base_url(self, website: str) -> str:
        """Normalize company website into a base https URL."""
        website = website.strip()
        if not website.startswith(("http://", "https://")):
            website = f"https://{website}"
        parsed = urlparse(website)
        return f"{parsed.scheme}://{parsed.netloc}"

    def _extract_title(self, html: str) -> str:
        """Extract HTML page title."""
        soup = BeautifulSoup(html, "lxml")
        if soup.title and soup.title.string:
            return soup.title.string.strip()
        return ""

    def _is_candidate_document_link(self, url: str, text: str) -> bool:
        """Filter links down to plausible IR documents and releases."""
        lower_text = text.lower()
        lower_url = url.lower()

        if any(
            token in lower_text or token in lower_url
            for token in [
                "annual",
                "earnings",
                "results",
                "presentation",
                "investor",
                "shareholder",
                "quarter",
                "financial",
                "10-k",
            ]
        ):
            return True

        return lower_url.endswith((".pdf", ".htm", ".html"))

    def _classify_link(self, text: str, url: str) -> Tuple[Optional[str], float]:
        """Classify a discovered link into one of the supported IR document types."""
        haystack = f"{text} {url}".lower()
        best_type = None
        best_score = 0.0

        for doc_type, keywords in self.CLASSIFICATION_RULES.items():
            score = 0
            for keyword in keywords:
                if keyword in haystack:
                    score += 1

            if score > best_score:
                best_type = doc_type
                best_score = float(score)

        if not best_type:
            return None, 0.0

        return best_type, best_score

    def _file_type_from_url(self, url: str) -> str:
        """Infer file type from URL."""
        path = urlparse(url).path.lower()
        if path.endswith(".pdf"):
            return "pdf"
        if path.endswith(".html") or path.endswith(".htm"):
            return "html"
        return "web"

    def _suffix_from_response(self, url: str, content_type: str) -> str:
        """Choose a file suffix when downloading a document."""
        lower_url = url.lower()
        lower_type = content_type.lower()
        if lower_url.endswith(".pdf") or "pdf" in lower_type:
            return ".pdf"
        if lower_url.endswith(".htm") or lower_url.endswith(".html") or "html" in lower_type:
            return ".html"
        return ".bin"

    def _filename_from_url(self, url: str) -> str:
        """Get a fallback filename stem from a URL."""
        path = urlparse(url).path.rstrip("/")
        if not path:
            return "document"
        return Path(path).stem or "document"

    def _safe_filename(self, value: str) -> str:
        """Convert free text into a filesystem-safe filename."""
        cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
        return cleaned[:100].strip("_") or "document"


if __name__ == "__main__":
    collector = IRDocumentCollector()

    # Demo mode: discover without downloading.
    result = collector.collect_company_documents("AAPL", download=False, max_documents=15)
    print(json.dumps(result, indent=2))
