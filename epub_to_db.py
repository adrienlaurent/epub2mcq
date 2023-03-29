# Copyright Adrien Laurent 2022
# All rights reserved.

from epub import get_chapters_from_epub
import firebase

firebase.init_firebase()
ref = firebase.db.reference('books')

chapters = get_chapters_from_epub("BOOK.epub")
ref.child("BOOK/CHAPTERS").set(chapters)
