.. _creating-a-custom-method:

Adding a new method
===================

.. code-block:: python

   from diversify_text import Diversifier
   from diversify_text.method import DiversificationMethod


   class MyMethod(DiversificationMethod):
       name = "my_method"

       def generate(self, texts, style_dict, *, max_new_tokens=None,
                    temperature=None, top_p=None, **kwargs):
           # style_dict maps each target style name to its example texts.
           return [[f"{text} :: {name}" for name in style_dict] for text in texts]


   results = Diversifier(method=MyMethod()).diversify(
       "Hello", styles=["scottish_english", "opinion"],
   )

.. code-block:: python

   [{"original": "Hello", "paraphrases": [
       {"style": "scottish_english", "text": "Hello :: scottish_english"},
       {"style": "opinion", "text": "Hello :: opinion"},
   ]}]

**Required:** a method must accept the two positional arguments —
``texts`` (the input texts) and ``style_dict`` (style name → example
texts) — and return one generated string per style for each text, in
``style_dict`` order (shape ``len(texts)`` x ``len(style_dict)``).
The style labels are attached by the core afterwards.

**Optional:** ``max_new_tokens``, ``temperature`` and ``top_p`` are
passed by the core; a method that has no use for them can ignore them.
Anything the caller provides via ``method_kwargs`` arrives as extra
keyword arguments, so method-specific options go there.
