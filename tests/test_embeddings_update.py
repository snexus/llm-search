import pandas as pd

from llmsearch.embeddings import get_changed_or_new_files


def test_new_only():
    new_hashes = pd.DataFrame(
        [
            {"filehash": "11111", "filename": "file1"},
            {"filehash": "22222", "filename": "file2"},
            {"filehash": "33333", "filename": "file3"},
        ]
    )
    existing_hashes = pd.DataFrame(
        [
            {"filehash": "11111", "filename": "file1"},
            {"filehash": "22222", "filename": "file2"},
        ]
    )

    changed_or_new, changed, deleted = get_changed_or_new_files(
        new_hashes, existing_hashes
    )
    assert len(changed_or_new) == 1
    assert len(changed) == 0
    assert len(deleted) == 0

    assert changed_or_new.iloc[0, 1] == "file3"


def test_delete_only():
    new_hashes = pd.DataFrame(
        [
            {"filehash": "11111", "filename": "file1"},
            {"filehash": "22222", "filename": "file2"},
        ]
    )
    existing_hashes = pd.DataFrame(
        [
            {"filehash": "11111", "filename": "file1"},
            {"filehash": "22222", "filename": "file2"},
            {"filehash": "33333", "filename": "file3"},
        ]
    )

    changed_or_new, changed, deleted = get_changed_or_new_files(
        new_hashes, existing_hashes
    )
    assert len(changed_or_new) == 0
    assert len(changed) == 0
    assert len(deleted) == 1

    assert deleted.iloc[0, 1] == "file3"


def test_change_only():
    new_hashes = pd.DataFrame(
        [
            {"filehash": "11111", "filename": "file1"},
            {"filehash": "22222", "filename": "file2"},
            {"filehash": "33331", "filename": "file3"},
            {"filehash": "44441", "filename": "file4"},
        ]
    )
    existing_hashes = pd.DataFrame(
        [
            {"filehash": "11111", "filename": "file1"},
            {"filehash": "22222", "filename": "file2"},
            {"filehash": "33333", "filename": "file3"},
            {"filehash": "44444", "filename": "file4"},
        ]
    )

    changed_or_new, changed, deleted = get_changed_or_new_files(
        new_hashes, existing_hashes
    )
    assert len(changed_or_new) == 2
    assert len(changed) == 2
    assert len(deleted) == 0

    assert changed_or_new.iloc[0, 1] == "file3"


def test_new_and_delete():
    new_hashes = pd.DataFrame(
        [
            {"filehash": "11111", "filename": "file1"},
            {"filehash": "22222", "filename": "file2"},
            {"filehash": "33333", "filename": "file3"},
            {"filehash": "55555", "filename": "file5"},
        ]
    )
    existing_hashes = pd.DataFrame(
        [
            {"filehash": "11111", "filename": "file1"},
            {"filehash": "22222", "filename": "file2"},
            {"filehash": "33333", "filename": "file3"},
            {"filehash": "44444", "filename": "file4"},
        ]
    )

    changed_or_new, changed, deleted = get_changed_or_new_files(
        new_hashes, existing_hashes
    )
    assert len(changed_or_new) == 1
    assert len(changed) == 0
    assert len(deleted) == 1

    assert changed_or_new.iloc[0, 1] == "file5"


def test_new_change_and_delete():
    new_hashes = pd.DataFrame(
        [
            {"filehash": "11111", "filename": "file1"},
            {"filehash": "22222", "filename": "file2"},
            {"filehash": "33333", "filename": "file3"},
            {"filehash": "55555", "filename": "file5"},
        ]
    )
    existing_hashes = pd.DataFrame(
        [
            {"filehash": "11111", "filename": "file1"},
            {"filehash": "22222", "filename": "file2"},
            {"filehash": "33332", "filename": "file3"},
            {"filehash": "44444", "filename": "file4"},
        ]
    )

    changed_or_new, changed, deleted = get_changed_or_new_files(
        new_hashes, existing_hashes
    )
    assert len(changed_or_new) == 2
    assert len(changed) == 1
    assert len(deleted) == 1

    assert changed.iloc[0, 1] == "file3"
    assert deleted.iloc[0, 1] == "file4"
