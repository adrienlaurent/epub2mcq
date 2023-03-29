# Copyright Adrien Laurent 2022
# All rights reserved.

"""
This script modifies the Firebase DB to add a new field called "nouns" in the
chunk, which contains the list of nouns found in the "context" text.
"""

import argparse
from typing import Dict, List, Union

import spacy

import firebase
from utils import filter_text


def get_nouns(nlp: spacy.language.Language, text: str) -> List[str]:
    """
    Get nouns from a text

    Args:
        nlp: Spacy model to use
        text: Text to process
    
    Returns:
        List of the nouns as strings
    """

    res = set()
    lower_res = set()
    doc = nlp(filter_text(text.encode("ascii", errors="ignore").decode("ascii")))
    for t in doc:
        if (
            t.text.lower() not in lower_res
            and (t.pos_ == "NOUN" or t.pos_ == "VERB" )
            and t.is_stop is False
            and t.text.isupper() is False
        ):
            res.add(t.text)
            lower_res.add(t.text.lower())
    return list(res)


def iterate(data: Union[List, Dict]):
    """
    Get an iterator that goes over (index, value) tuples in the data, where
    index and value are dependent on the type of data.
     - For dict: index = key (whatever type), value = data[index]
     - For list: index = index (int), value = data[index]
    """

    if isinstance(data, dict):
        return data.items()
    elif isinstance(data, list):
        return enumerate(data)
    else:
        raise RuntimeError(f"Data must be either a dict or list, not {type(data)}")


def process_book(
    book_dict: Dict,
    book_name: str,
    books_ref,
    nlp: spacy.language.Language,
    overwrite: bool = False,
):
    """
    Args:
        book_dict: Dictionary containing the book data
        book_name: Name of the book
        books_ref: Reference to the Firebase RTDB that contains the books
        nlp: Spacy model to use
        overwrite: Whether to overwrite the existing nouns field in the book
    """

    print(f"Processing book '{book_name}'")
    for chapter_num, chapter in iterate(book_dict["chapters"]):
        for chunk_num, chunk in iterate(chapter["chunks"]):
            if chunk.get("context") is None:
                print(f"No context found in chunk {chapter_num}/{chunk_num}")
                continue
            if overwrite is False and "nouns" in chunk:
                print(f"Nouns already exist in chunk {chapter_num}/{chunk_num}")
                continue
            else:
                nouns = get_nouns(nlp, chunk["context"])
                print(f"Found {len(nouns)} nouns in {chapter_num}/{chunk_num}")
                nouns_ref = books_ref.child(
                    f"{book_name}/chapters/{chapter_num}/chunks/{chunk_num}/nouns"
                )
                nouns_ref.set(nouns)
    print()


def main(args):
    nlp = spacy.load("en_core_web_sm")

    firebase.init_firebase()
    books_ref = firebase.db.reference("books")

    if args.bookname is not None:
        book = books_ref.child(args.bookname).get()
        if not book:
            raise RuntimeError(f"Could not find book {args.bookname} in the database")
        process_book(book, args.bookname, books_ref, nlp, overwrite=args.overwrite)
    else:
        print(
            "No book name provided (using --bookname), processing "
            "all books in Firebase RTDB\n"
        )
        books_data = books_ref.get()
        print(f"Found {len(books_data)} book(s)")
        for book_name, book in iterate(books_data):
            process_book(book, book_name, books_ref, nlp, overwrite=args.overwrite)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="This script modifies the Firebase DB to add a new field "
        "called 'nouns' in the chunk, which contains the list of nouns found "
        "in the 'context' text."
    )
    ap.add_argument(
        "--bookname",
        type=str,
        default=None,
        help="Name of the book to process, optionsl. All books will be "
        "processed if not provided",
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Whether to overwrite the nouns if they exist. Default: False",
    )
    args = ap.parse_args()
    main(args)
