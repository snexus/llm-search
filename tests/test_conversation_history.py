from llmsearch.config import ConversrationHistorySettings


def test_max_pairs():
    c = ConversrationHistorySettings(max_history_length=2, rewrite_query=False)

    c.add_qa_pair("question1", "answer1")
    c.add_qa_pair("question2", "answer2")
    c.add_qa_pair("question3", "answer3")

    assert len(c.history) == 2

    assert c.history[0].question == "question2"
    assert c.history[0].answer == "answer2"

    assert c.history[1].question == "question3"
    assert c.history[1].answer == "answer3"


def test_no_pairs():
    c = ConversrationHistorySettings(max_history_length=2, rewrite_query=False)
    assert c.chat_history == ""
