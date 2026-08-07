"""Instruction bank for the zero_shot method.

Each entry is one style, defined by a single rewrite instruction for
the causal language model (instead of example texts).  An instruction
may place the input text itself via the ``[DOCUMENT SEGMENT]``
placeholder; otherwise the text is appended at the end during
generation.
"""

# Bank order matters: n selects the first n styles, so the best-curated
# styles come first.
ZERO_SHOT_STYLE_BANK: dict[str, list[str]] = {
    "formal": [  # taken from https://arxiv.org/abs/2109.03910
        "Here is some text: [DOCUMENT SEGMENT]. Output only the rewrite, nothing else. Here is a rewrite of the text, which is more formal:"
    ],
    "simple": [  # taken from https://aclanthology.org/2023.emnlp-main.821.pdf
        "Please rewrite the following complex sentence in order to "
        "make it easier to understand by non-native speakers of English. "
        "You can do so by replacing complex words with simpler "
        "synonyms (i.e.paraphrasing), deleting unimportant information "
        "(i.e.compression), and/or splitting a long complex sentence into "
        "several simpler ones. The final simplified sentence needs to be "
        "grammatical, fluent, and retain the main ideas of its original "
        "counterpart without altering its meaning. Output only the paraphrase, nothing else. "
        "Text: [DOCUMENT SEGMENT]"
    ],
    "complex": [
        "Please rewrite the following simple sentence in order to "
        "make it more complex and sophisticated. "
        "You can do so by replacing simple words with more elaborate "
        "synonyms (i.e. paraphrasing), adding relevant detail and nuance "
        "(i.e. expansion), and/or combining several short sentences into "
        "longer, more intricate ones. The final complex sentence needs to be "
        "grammatical, fluent, and retain the main ideas of its original "
        "counterpart without altering its meaning. Output only the rewrite, "
        "nothing else. "
        "Text: [DOCUMENT SEGMENT]"
    ],
    "caps": [
        "Here is some text: [DOCUMENT SEGMENT]. Output only the rewrite, "
        "nothing else. Here is a rewrite of the text, which is in ALL CAPS:"
    ],
    "lowercase": [
        "Here is some text: [DOCUMENT SEGMENT]. Output only the rewrite, "
        "nothing else. Here is a rewrite of the text, which is in all "
        "lower case:"
    ],
    "text_emojis": [
        "Here is some text: [DOCUMENT SEGMENT]. Output only the rewrite, "
        "nothing else. Here is a rewrite of the text, which includes "
        "text emojis like :-) or ;-D:"
    ],
    "less_common_verbs": [
        "Here is some text: [DOCUMENT SEGMENT]. Output only the rewrite, "
        "nothing else. Here is a rewrite of the text, which uses less "
        "common verbs:"
    ],
    "wikipedia": [  # taken from https://arxiv.org/abs/2401.16380
        "For the following paragraph give me a diverse paraphrase of the same "
        "in high quality English language as in sentences on Wikipedia. "
        "Output only the paraphrase, nothing else. "
        "Text: [DOCUMENT SEGMENT]"
    ],
    # From: https://arxiv.org/abs/2401.05952 (Zhang et al., 2024)
    # Original 5-modification prompt from the paper (no emojis).
    "humanize": [
        "I need to modify a machine-generated text to make it appear more like it was "
        "written by a human. The objective is to introduce elements commonly found in "
        "human-written texts. Here are some optional modifications you can choose to "
        "apply:\n"
        "1. Introduce spelling errors or typos (optional).\n"
        "2. Create grammatical errors, such as randomly adding or deleting words "
        "(optional).\n"
        "3. Include relevant but internet links, like blog posts or image links pertaining "
        "to the topic, you don't have to use the real links, so you can freely write one "
        "(optional).\n"
        "4. Add relevant hashtags, for instance, #TopicKeyword #Location #Activity "
        "(optional).\n"
        "5. Use internet slang and abbreviations, e.g., 'OMG', 'How r u', 'LOL' "
        "(optional).\n"
        "Please select any combination of these modifications to enhance the text's "
        "human-like quality. The aim is to simulate the imperfections and stylistic "
        "choices typical in casual human writing.\n"
        "The word count of the new text should not exceed 1.1 times that of the original "
        "text.\n"
        "You should just give me the revised version without any other words.\n"
        "Emojis are strictly prohibitive, so please ensure that it contains no emojis.\n"
        "Here is the machine-generated text: [DOCUMENT SEGMENT]"
    ],
    # -----------------------------------------------
    # The following instructions are taken from the finephrase space:
    # https://huggingface.co/spaces/HuggingFaceFW/finephrase
    #       --> actually they are not so great templates for REPHRASING,
    #       they will often hallucinate new information to match the
    #       target register
    # ------------------------------------------------
    "article": [
        "Transform the document into a magazine-style feature article. "
        "Open with an engaging lead, then blend narrative storytelling with "
        "factual explanation. Maintain an accessible yet polished tone suitable "
        "for a general but informed readership. "
        "Output only the feature article, nothing else. "
        "Document: [DOCUMENT SEGMENT]"
    ],
    "commentary": [
        "Summarize the document in a concise paragraph that captures its central "
        "arguments or findings. Then, write an expert commentary that critically "
        "reflects on its implications, limitations, or broader context. Maintain "
        "an analytical and professional tone throughout. "
        "Output only the summary and the commentary, nothing else. "
        "Document: [DOCUMENT SEGMENT]"
    ],
    "discussion": [
        "Reformulate the document as a dialogue between a teacher and a student. "
        "The teacher should guide the student toward understanding the key points "
        "while clarifying complex concepts. Keep the exchange natural, informative, "
        "and faithful to the original content. "
        "Output only the dialogue, nothing else. "
        "Document: [DOCUMENT SEGMENT]"
    ],
    "explanation": [
        "Rewrite the document to provide clear scientific or logical explanations "
        "for concepts, phenomena, or processes mentioned in the text. Make implicit "
        "reasoning explicit by explaining why things work the way they do, what "
        "principles or mechanisms are at play, and how different factors relate to "
        "each other. Focus on building understanding through causal explanations "
        "rather than just describing facts. "
        "Output only the explanatory text, nothing else. "
        "Document: [DOCUMENT SEGMENT]"
    ],
    "faq": [
        "Rewrite the document as a comprehensive FAQ (Frequently Asked Questions). "
        "Extract or infer the key questions a reader would have about this topic, "
        "then provide clear, direct answers. Order questions logically—from "
        "foundational to advanced, or by topic area. Each answer should be "
        "self-contained and understandable without reference to other answers. "
        "Ensure the FAQ works as a standalone document. "
        "Output only the FAQ, nothing else. "
        "Document: [DOCUMENT SEGMENT]"
    ],
    "math": [
        "Rewrite the document to create a mathematical word problem based on the "
        "numerical data or relationships in the text. Provide a step-by-step "
        "solution that shows the calculation process clearly. Create a problem "
        "that requires multi-step reasoning and basic arithmetic operations. It "
        "should include the question followed by a detailed solution showing each "
        "calculation step. "
        "Output only the problem and solution, nothing else. "
        "Document: [DOCUMENT SEGMENT]"
    ],
    "narrative": [
        "Rewrite the document as a clear narrative that emphasizes the temporal "
        "sequence and causal relationships between events or steps. Reorganize "
        "the content to show how actions, events, or situations naturally flow "
        "from one to the next, making cause-and-effect relationships explicit. "
        "If describing a process or activity, show the logical progression of "
        "steps and explain why each step follows from the previous one. "
        "Output only the narrative, nothing else. "
        "Document: [DOCUMENT SEGMENT]"
    ],
    "table": [
        "Rewrite the document as a structured table that organizes the key "
        "information, then generate one question-answer pair based on the table. "
        "First extract the main data points and organize them into a clear table "
        "format with appropriate headers using markdown table syntax with proper "
        "alignment. After the table, generate one insightful question that can be "
        "answered using the table data. Provide a clear, concise answer to the "
        "question based on the information in the table. "
        "Output only the table followed by the question-answer pair, nothing else. "
        "Document: [DOCUMENT SEGMENT]"
    ],
    "tutorial": [
        "Rewrite the document as a clear, step-by-step tutorial or instructional "
        "guide. Use numbered steps or bullet points where appropriate to enhance "
        "clarity. Preserve all essential information while ensuring the style "
        "feels didactic and easy to follow. "
        "Output only the tutorial, nothing else. "
        "Document: [DOCUMENT SEGMENT]"
    ],
}
