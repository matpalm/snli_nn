def split_binary_parse(parse, include_parenthesis):
    tokens = []
    for token in parse.split(" "):
        if token == "(" or token == ")":
            if include_parenthesis:
                tokens.append(token)
        else:
            tokens.append(token.lower())
    return tokens

# split a parse e.g. "(ROOT (NP (DT A) (JJ young)))"
# into sequence of open tags, lower case tokens, close tags (and drop ROOT)
# e.g. (NP (DT a DT) (JJ young JJ) NP)
def split_parse_with_open_close(s):
    tag_stack = []  # stack of tags required for closing in reverse order
    tokens = []  # open / close tags + tokens
    elems = s.split(" ")
    for e in elems:
        if e.startswith("("):
            # opening tag, eg "(NP", add "(NP" to tokens and "NP" to stack
            tokens.append(e)
            tag_stack.append(e[1:])
        else:
            assert e.endswith(")"), e
            closing_tags = []
            while e.endswith(")"):
                assert tag_stack, tag_stack
                closing_tags.append(tag_stack.pop() + ")")
                e = e[:-1]
            tokens.append(e.lower())
            tokens.extend(closing_tags)

    assert not tag_stack, tag_stack
    assert tokens[0] == "(ROOT"
    assert tokens[-1] == "ROOT)"
    return tokens[1:-1]

def tokens_for(eg, sentence_idx, parse_mode):
    if parse_mode in ["PARSE_WITH_OPEN_CLOSE_TAGS", "JUST_OPEN_CLOSE_TAGS"]:
        parse_field = "sentence%d_parse" % sentence_idx
    else:
        parse_field = "sentence%d_binary_parse" % sentence_idx
    parse_str = eg[parse_field]

    if parse_mode == "BINARY_WITHOUT_PARENTHESIS":
        # "a person by a car"
        return split_binary_parse(parse_str, include_parenthesis=False) 

    elif parse_mode == "BINARY_WITH_PARENTHESIS":
        # "( ( a person ) ( by ( a car ) ) )"
        return split_binary_parse(parse_str, include_parenthesis=True)

    elif parse_mode == "PARSE_WITH_OPEN_CLOSE_TAGS":
        # "(NP (NP (DT a DT) (NN person NN) NP) (PP (IN by IN) (NP (DT a DT) (NN car NN) NP) PP) NP) NP)"
        return split_parse_with_open_close(parse_str)

    elif parse_mode == "JUST_OPEN_CLOSE_TAGS":
        # "(NP (NP (DT DT) (NN NN) NP) (PP (IN IN) (NP (DT DT) (NN NN) NP) PP) NP) NP)"
        return filter(lambda t: t.startswith("(") or t.endswith(")"),
                      split_parse_with_open_close(parse_str))

    else:
        raise Exception("unknown parse-mode [%s] ; expected BINARY_WITHOUT_PARENTHESIS|BINARY_WITH_PARENTHESIS|PARSE_WITH_OPEN_CLOSE_TAGS" % parse_mode)

