;; Information Retrieval
;; Packaged on 2023-3-24

(in-package :cl-user)

(defpackage :information-retrieval
  (:use :cl :py4cl)
  (:export
    init
    embed-documents
    retrieve
    rerank
    retrieve+rerank))