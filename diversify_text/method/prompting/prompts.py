"""Prompt bank for the method prompting (L)LMs to diversify text."""

from __future__ import annotations

# -- Prompt key constants -----------------------------------------------------
# Used in both the bank dict and DEFAULT_PROMPTS so renaming a key only
# requires a single change.

WIKIPEDIA_PARAPHRASE = "wikipedia_paraphrase"
FINEPHRASE_ARTICLE = "finephrase_article"
FINEPHRASE_COMMENTARY = "finephrase_commentary"
FINEPHRASE_DISCUSSION = "finephrase_discussion"
FINEPHRASE_EXPLANATION = "finephrase_explanation"
FINEPHRASE_FAQ = "finephrase_faq"
FINEPHRASE_MATH = "finephrase_math"
FINEPHRASE_NARRATIVE = "finephrase_narrative"
FINEPHRASE_TABLE = "finephrase_table"
FINEPHRASE_TUTORIAL = "finephrase_tutorial"
HUMANIZE_LLM_AS_COAUTHOR_ORIGINAL = "humanize_llm-as-coauthor_original"
FORMAL_REIF = "formal_reif"
SIMPLE_REIF = "simple_reif"
SIMPLE_KEW = "simple_kew"
COMPLEX_KEW = "complex_kew"
PASSIVE_REIF = "passive_reif"
CAPS_REIF = "caps_reif"
LOWCAPS_REIF = "lowcaps_reif"
TEXT_EMOJIS_REIF = "text_emojis_reif"
LESS_COMMON_VERBS_REIF = "less_common_verbs_reif"

# -- Zero-shot style transfer templates ---------------------------------------
#   These are prompts taken from other work that aim to rewrite a text in a given style/structure.
#   Uses only [DOCUMENT SEGMENT] placeholders.

ZS_PROMPT_BANK: dict[str, str] = {
    WIKIPEDIA_PARAPHRASE: ( # taken from https://arxiv.org/abs/2401.16380
        "For the following paragraph give me a diverse paraphrase of the same "
        "in high quality English language as in sentences on Wikipedia. "
        "Output only the paraphrase, nothing else. "
        "Text: [DOCUMENT SEGMENT]"
    ),
    FORMAL_REIF: ( # taken from https://arxiv.org/abs/2109.03910
        "Here is some text: [DOCUMENT SEGMENT]. Output only the rewrite, nothing else. Here is a rewrite of the text, which is more formal:"
    ),
    SIMPLE_REIF: ( # taken from https://arxiv.org/abs/2109.03910
        "Here is some text: [DOCUMENT SEGMENT]. Output only the rewrite, nothing else. Here is a rewrite of the text, which is more simple:"
    ),
    CAPS_REIF: (
        "Here is some text: [DOCUMENT SEGMENT]. Output only the rewrite, "
        "nothing else. Here is a rewrite of the text, which is in ALL CAPS:"
    ),
    LOWCAPS_REIF: (
        "Here is some text: [DOCUMENT SEGMENT]. Output only the rewrite, "
        "nothing else. Here is a rewrite of the text, which is in all "
        "lower case:"
    ),
    TEXT_EMOJIS_REIF: (
        "Here is some text: [DOCUMENT SEGMENT]. Output only the rewrite, "
        "nothing else. Here is a rewrite of the text, which includes "
        "text emojis like :-) or ;-D:"
    ),
    LESS_COMMON_VERBS_REIF: (
        "Here is some text: [DOCUMENT SEGMENT]. Output only the rewrite, "
        "nothing else. Here is a rewrite of the text, which uses less "
        "common verbs:"
    ),
    SIMPLE_KEW: ( # taken from https://aclanthology.org/2023.emnlp-main.821.pdf
        "Please rewrite the following complex sentence in order to "
        "make it easier to understand by non-native speakers of English. "
        "You can do so by replacing complex words with simpler "
        "synonyms (i.e.paraphrasing), deleting unimportant information "
        "(i.e.compression), and/or splitting a long complex sentence into "
        "several simpler ones. The final simplified sentence needs to be "
        "grammatical, fluent, and retain the main ideas of its original "
        "counterpart without altering its meaning. Output only the paraphrase, nothing else. "
        "Text: [DOCUMENT SEGMENT]"
    ),
    COMPLEX_KEW: (
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
    ),
    # From: https://arxiv.org/abs/2401.05952 (Zhang et al., 2024)
    # Original 5-modification prompt from the paper (no emojis).
    HUMANIZE_LLM_AS_COAUTHOR_ORIGINAL: (
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
    ),
    # -----------------------------------------------
    # The following prompts are taken from the finephrase space: https://huggingface.co/spaces/HuggingFaceFW/finephrase
    #       --> acutally they are not so great templates for REPHRASING,
    #       they will often hallucinate new information to match the target register
    # ------------------------------------------------
    FINEPHRASE_ARTICLE: ( # taken from https://huggingface.co/spaces/HuggingFaceFW/finephrase
        "Transform the document into a magazine-style feature article. "
        "Open with an engaging lead, then blend narrative storytelling with "
        "factual explanation. Maintain an accessible yet polished tone suitable "
        "for a general but informed readership. "
        "Output only the feature article, nothing else. "
        "Document: [DOCUMENT SEGMENT]"
    ),
    FINEPHRASE_COMMENTARY: ( # taken from https://huggingface.co/spaces/HuggingFaceFW/finephrase
        "Summarize the document in a concise paragraph that captures its central "
        "arguments or findings. Then, write an expert commentary that critically "
        "reflects on its implications, limitations, or broader context. Maintain "
        "an analytical and professional tone throughout. "
        "Output only the summary and the commentary, nothing else. "
        "Document: [DOCUMENT SEGMENT]"
    ),
    FINEPHRASE_DISCUSSION: ( # taken from https://huggingface.co/spaces/HuggingFaceFW/finephrase
        "Reformulate the document as a dialogue between a teacher and a student. "
        "The teacher should guide the student toward understanding the key points "
        "while clarifying complex concepts. Keep the exchange natural, informative, "
        "and faithful to the original content. "
        "Output only the dialogue, nothing else. "
        "Document: [DOCUMENT SEGMENT]"
    ),
    FINEPHRASE_EXPLANATION: ( # taken from https://huggingface.co/spaces/HuggingFaceFW/finephrase
        "Rewrite the document to provide clear scientific or logical explanations "
        "for concepts, phenomena, or processes mentioned in the text. Make implicit "
        "reasoning explicit by explaining why things work the way they do, what "
        "principles or mechanisms are at play, and how different factors relate to "
        "each other. Focus on building understanding through causal explanations "
        "rather than just describing facts. "
        "Output only the explanatory text, nothing else. "
        "Document: [DOCUMENT SEGMENT]"
    ),
    FINEPHRASE_FAQ: ( # taken from https://huggingface.co/spaces/HuggingFaceFW/finephrase
        "Rewrite the document as a comprehensive FAQ (Frequently Asked Questions). "
        "Extract or infer the key questions a reader would have about this topic, "
        "then provide clear, direct answers. Order questions logically—from "
        "foundational to advanced, or by topic area. Each answer should be "
        "self-contained and understandable without reference to other answers. "
        "Ensure the FAQ works as a standalone document. "
        "Output only the FAQ, nothing else. "
        "Document: [DOCUMENT SEGMENT]"
    ),
    FINEPHRASE_MATH: ( # taken from https://huggingface.co/spaces/HuggingFaceFW/finephrase
        "Rewrite the document to create a mathematical word problem based on the "
        "numerical data or relationships in the text. Provide a step-by-step "
        "solution that shows the calculation process clearly. Create a problem "
        "that requires multi-step reasoning and basic arithmetic operations. It "
        "should include the question followed by a detailed solution showing each "
        "calculation step. "
        "Output only the problem and solution, nothing else. "
        "Document: [DOCUMENT SEGMENT]"
    ),
    FINEPHRASE_NARRATIVE: ( # taken from https://huggingface.co/spaces/HuggingFaceFW/finephrase
        "Rewrite the document as a clear narrative that emphasizes the temporal "
        "sequence and causal relationships between events or steps. Reorganize "
        "the content to show how actions, events, or situations naturally flow "
        "from one to the next, making cause-and-effect relationships explicit. "
        "If describing a process or activity, show the logical progression of "
        "steps and explain why each step follows from the previous one. "
        "Output only the narrative, nothing else. "
        "Document: [DOCUMENT SEGMENT]"
    ),
    FINEPHRASE_TABLE: ( # taken from https://huggingface.co/spaces/HuggingFaceFW/finephrase
        "Rewrite the document as a structured table that organizes the key "
        "information, then generate one question-answer pair based on the table. "
        "First extract the main data points and organize them into a clear table "
        "format with appropriate headers using markdown table syntax with proper "
        "alignment. After the table, generate one insightful question that can be "
        "answered using the table data. Provide a clear, concise answer to the "
        "question based on the information in the table. "
        "Output only the table followed by the question-answer pair, nothing else. "
        "Document: [DOCUMENT SEGMENT]"
    ),
    FINEPHRASE_TUTORIAL: ( # taken from https://huggingface.co/spaces/HuggingFaceFW/finephrase
        "Rewrite the document as a clear, step-by-step tutorial or instructional "
        "guide. Use numbered steps or bullet points where appropriate to enhance "
        "clarity. Preserve all essential information while ensuring the style "
        "feels didactic and easy to follow. "
        "Output only the tutorial, nothing else. "
        "Document: [DOCUMENT SEGMENT]"
    ),
}

# -- Example-basd style transfer templates ---------------------------------------
#   These are prompts created by us or inspired by other work that rewrite a text with examples in the target style.
#   The style/structure itself is not described in the prompt as for the zero-shot templates.
#   These use [STYLE EXAMPLES], [STYLE NAME] and [DOCUMENT SEGMENT] placeholders.

EXAMPLE_BASED_PROMPT_BANK: dict[str, str] = {
    "style_transfer": (  # ours
        "Here are examples of the [STYLE NAME] writing style:\n"
        "[STYLE EXAMPLES]\n\n"
        "For the following document give me a diverse paraphrase of the same "
        "that matches the human [STYLE NAME] writing style demonstrated in the examples above. "
        "Preserve the same information. "
        "Output only the paraphrase, nothing else.\n"
        "Document: [DOCUMENT SEGMENT]"
    ),
    "humanize_transfer": (  # inspired by  https://arxiv.org/abs/2401.05952
        "I need to modify a machine-generated text to make it appear more like it was "
        "written by a human. The objective is to introduce elements found in "
        "human-written texts in the [STYLE NAME] style. Here are some examples of "
        "human written texts in [STYLE NAME] style:\n"
        "[STYLE EXAMPLES]\n\n"
        "Please select any combination of these modifications to enhance the text's "
        "human-like quality. The aim is to simulate the imperfections and stylistic "
        "choices typical in the exemplified [STYLE NAME] human style.\n"
        "The word count of the new text should not exceed 1.1 times that of the original "
        "text.\n"
        "You should just give me the revised version without any other words.\n"
        "Here is the machine-generated text: [DOCUMENT SEGMENT]"
    ),
}

# -- Name-based style transfer templates ------------------------------------------------------

NAME_BASED_PROMPT_BANK: dict[str, str] = {
    "reif": (
        "Here is some text: [DOCUMENT SEGMENT]. Here is a rewrite of the text, which is more [STYLE NAME]:"
    ),
}


# Placeholder tokens used inside prompt templates.
PLACEHOLDER_TEXT = "[DOCUMENT SEGMENT]"
PLACEHOLDER_STYLE_EXAMPLES = "[STYLE EXAMPLES]"
PLACEHOLDER_STYLE_NAME = "[STYLE NAME]"

PROMPT_BANK: dict[str, str] = {**ZS_PROMPT_BANK, **EXAMPLE_BASED_PROMPT_BANK, **NAME_BASED_PROMPT_BANK}

#: Prompt keys whose templates depend on style examples or style names.
STYLE_DEP_PROMPTS: set[str] = {*EXAMPLE_BASED_PROMPT_BANK, *NAME_BASED_PROMPT_BANK}

DEFAULT_PROMPTS: list[str] = [
    HUMANIZE_LLM_AS_COAUTHOR_ORIGINAL,
    COMPLEX_KEW,
    SIMPLE_KEW,
    WIKIPEDIA_PARAPHRASE,
    CAPS_REIF,
]