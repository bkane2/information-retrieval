Information Retrieval
=======

A package for doing basic information retrieval using Python's SentenceTransformer library.

## Dependencies
- Quicklisp
- [ASDF version 3 or above](https://common-lisp.net/project/asdf/archives/asdf.lisp)
- [sentence_transformers](https://pypi.org/project/sentence-transformers/), Install with `pip install sentence_transformers`.
- HuggingFace [transformers](https://huggingface.co/docs/transformers/installation), Install with `pip install transformers`.
- [numpy](https://numpy.org/)
- [pandas](https://pandas.pydata.org/)
- py4cl (loaded automatically via quicklisp)

The current version of the code has only been tested on SBCL on Linux.

If difficulties are encountered using the library on Windows, you may need to install [py4cl2](https://github.com/digikar99/py4cl2) locally, and set the `pycmd` config variable for py4cl2 to your `python.exe` path (see `*config* / config-var` in the [documentation](https://digikar99.github.io/py4cl2/) to see how to do this). In this case, you will need to swap out all py4cl function calls with the corresponding py4cl2 function calls in the code (e.g., replace `py4cl:python-exec` with `py4cl2:pyexec`).

## Installation
1. Install quicklisp by following instructions at https://www.quicklisp.org/beta/
2. Download the latest [asdf.lisp](https://common-lisp.net/project/asdf/#downloads) file and include it in your lisp start-up script (e.g. `.sbclrc`). I recommend also overwriting `quicklisp/asdf.lisp` to eliminate the possibility of accidentally loading the out-of-date copy of `asdf.lisp` that comes with Quicklisp be default.
3. It's best to pre-download the sentence_transformers models you wish to use using the appropriate library functions.

## Using the package

### Initialization
Initialize the package to pre-load the models using `(information-retrieval:init)`. To specify which sentence_transformer models to use for the [candidate retrieval](https://www.sbert.net/docs/pretrained_models.html) and [cross-encoder re-scoring](https://huggingface.co/cross-encoder) (if used), set the model names beforehand:
```lisp
$ sbcl
$ (ql:quickload :information-retrieval)
$ ...[loading messages]...
$ (information-retrieval:set-model "all-distilroberta-v1")
$ (information-retrieval:set-cross-encoder "cross-encoder/ms-marco-electra-base")
$ (information-retrieval:init)
```
Note that the two models above are the defaults used if initializing the package without specifying models explicitly.

The package also supports use of the HuggingFace API for candidate retrieval (the cross-encoder model is currently not supported). To use the API, set the *api* flag before initialization (and make sure that the correct API model path is given):

```lisp
$ sbcl
$ (ql:quickload :information-retrieval)
$ ...[loading messages]...
$ (information-retrieval:set-api t)
$ (information-retrieval:set-model "sentence-transformers/all-distilroberta-v1")
$ (information-retrieval:init :api-key *api-key*)
```

The API key will also need to be provided as a keyword arg to the other functions (see below).

### Embedding documents
The `embed-documents` function allows for a list of documents (i.e., strings) to be encoded as embeddings using the chosen sentence_transformer model:
```lisp
$ (setq documents '("Sentence one." "Sentence two." "Sentence three."))
$ (information-retrieval:embed-documents documents)
```

By default, this will return a list of Lisp vectors. The documents and embeddings can also be written to a specified CSV file using the optional `:filename` keyword argument, or appended to an existing CSV file by additionally specifying `:append t`:

```lisp
$ (setq documents '("Sentence one." "Sentence two." "Sentence three."))
$ (information-retrieval:embed-documents documents :filename "test.csv" :append t)
```

If using API, provide the API key as a keyword argument (this function will not actually embed the documents, but will still output
a CSV file containing the documents if a filename is specified):

```lisp
$ (setq documents '("Sentence one." "Sentence two." "Sentence three."))
$ (information-retrieval:embed-documents documents :filename "test.csv" :api-key *api-key*)
```

### Retrieving candidates
The `retrieve` function is used to select the `n` top candidates using the chosen sentence_transformer model. The function must be called providing either a name of a CSV file containing document and embedding columns, a list of document+embedding pairs, or a list of documents to embed using the model. The function returns a list of the `n` most similar strings (5 by default).

```lisp
$ (setq text '("Test sentence."))
$ (information-retrieval:retrieve text :n 5 :filename "test.csv")
```

```lisp
$ (setq text '("Test sentence."))
$ (information-retrieval:retrieve text :n 5 :documents+embeddings '(("Sentence one." #(...)) ("Sentence two." #(...))))
```

```lisp
$ (setq text '("Test sentence."))
$ (information-retrieval:retrieve text :n 5 :documents '("Sentence one." "Sentence two."))
```

If using API, provide the API key as a keyword argument:

```lisp
$ (setq text '("Test sentence."))
$ (information-retrieval:retrieve text :n 5 :filename "test.csv" :api-key *api-key*)
```


### Reranking candidates
The `rerank` function is used to select the final `n` top documents (1 by default) from a given list of documents using the chosen cross-encoder model.

```lisp
$ (setq text '("Test sentence."))
$ (information-retrieval:rerank text :n 1 :documents '("Sentence one." "Sentence two."))
```

API mode is currently not supported for this function.


### Retrieving and reranking
The `retrieve+rerank` function combines `retrieve` and `rerank` to select a list of the `n-candidate` most similar documents, and then the `n-result` top documents after reranking. The other keyword arguments should be the same as the `retrieve` function.

```lisp
$ (setq text '("Test sentence."))
$ (information-retrieval:retrieve+rerank text :n-candidate 5 :n-result 1 :filename "test.csv")
```

API mode is currently not supported for this function.