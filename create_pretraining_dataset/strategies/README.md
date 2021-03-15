# Strategies

Strategies are objects that are called through the classic `__call__` method. They receive in input a sequence of documents and must return a list of one or many created examples. The length of the output is not required to be the same of the input. In fact, usually a single input documents may create many output examples.

Strategies are instantiated by providing them with the `Namespace` of parameters collected from the command line and with the actual `tokenizer` instance.