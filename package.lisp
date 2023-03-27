;; Information Retrieval
;; Packaged on 2023-3-24

(in-package :cl-user)

(defpackage :information-retrieval
  (:use :cl :py4cl)
  (:export
    set-model
    set-cross-encoder
    init
    embed-documents
    retrieve
    rerank
    retrieve+rerank))