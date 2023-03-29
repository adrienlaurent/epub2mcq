import ebooklib
from ebooklib import epub
from nltk import sent_tokenize
from bs4 import BeautifulSoup, Tag

import config
from utils import filter_text


def get_chunks_from_paragraphs(paragraphs):
    """
    Split into chunks of sentences where each chunk is greater than
    config.MIN_CHUNK_LENGTH and less than config.MAX_CHUNK_LENGTH
    """

    chunks = []
    cur_chunk = ""

    # merge paragraphs into text and split into sentences
    text = " ".join(map(filter_text, paragraphs))
    sentences = sent_tokenize(text)

    # go over paragraphs, add a new chunk when it's size is greater than
    # MIN_CHUNK_LEN, but adding next paragraph would exceed MAX_CHUNK_LEN
    for sentence in sentences:
        candidate_chunk = cur_chunk + " " * bool(cur_chunk) + sentence
        if (
            len(cur_chunk) > config.MIN_CHUNK_LEN
            and len(candidate_chunk) > config.MAX_CHUNK_LEN
        ):
            chunks.append(cur_chunk)
            cur_chunk = ""
            candidate_chunk = sentence
        cur_chunk = candidate_chunk

    # add remaining chunk if its length is okay
    if config.MIN_CHUNK_LEN < len(cur_chunk) < config.MAX_CHUNK_LEN:
        chunks.append(cur_chunk)

    return chunks


def _process_paragraph(para: Tag) -> str:

    # if non-ASCII characters are found, ignore the paragraph
    try:
        para.get_text().encode("ascii")
    except UnicodeEncodeError:
        return None

    text = para.get_text().replace(u"\xa0", u" ")

    # add bold tag if it exists & is long enough, because
    # some books also contain subtitles in <b> tags
    if para.b and len(para.b.get_text()) > 3:
        b_text = para.b.get_text()
        text = text.replace(b_text, "<b>" + b_text + "</b>\n")
    
    return text.strip()


def chapter_to_chunks(chapter):
    soup = BeautifulSoup(chapter.get_body_content(), "html.parser")

    # get & process paragraphs
    paragraphs = list(
        filter(None, (_process_paragraph(para) for para in soup.find_all("p")))
    )
    chunks = get_chunks_from_paragraphs(paragraphs)

    # NOTE: the headers' content differs from book to book, therefore
    # we try to get text from h3 tags, but if it's not found,
    # we use the h2 tag instead, and if that's not found, we use the h1 tag
    # the empty title is returned when none of the header tags are found

    title = ""
    h3 = soup.find("h3")
    h2 = soup.find("h2")
    h1 = soup.find("h1")

    if h3:
        title = h3.get_text()
    elif h2:
        title = h2.get_text()
    elif h1:
        title = h1.get_text()

    # if title is not empty, prepend it to the first chunk with bold tags
    if title:
        chunks[0] = "<b>" + title + "</b>" + "\n" + chunks[0]

    return title, chunks


def get_chapters_from_epub(filepath):
    book = epub.read_epub(filepath)
    items = list(book.get_items_of_type(ebooklib.ITEM_DOCUMENT))
    chapters = [
        {"title": title.strip(), "chunks": chunks}
        for title, chunks in filter(
            lambda c: len(" ".join(c[1]).strip()) > 10,
            map(chapter_to_chunks, items)
        )
    ]
    return chapters