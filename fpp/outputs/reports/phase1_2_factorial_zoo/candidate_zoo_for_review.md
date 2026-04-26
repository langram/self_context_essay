# Factorial input zoo — candidate review

Each row is one of the 24 candidate inputs for Phase 1.2 Experiment H. 
Verify that within each (content) group of 8, factor flips are clean — i.e. the 'capital' vs 'lowercase' pair within fixed (markup, punct) differs only in case, etc.

Columns:
- `case_check`: ✅ if `starts_with_capital_letter` matches the declared `case` factor
- `markup_check`: ✅ if `contains_markup` matches the declared `markup` factor
- `punct_check`: shows the actual punct-token count; verify the high/low factor declaration is reasonable

| idx | content | case | markup | punct | n_tok | punct_tok | case_check | markup_check | text |
| ---: | --- | --- | --- | --- | ---: | ---: | --- | --- | --- |
| 0 | natural | capital | with | high | 28 | 9 | ✅ | ✅ | `<p>The cat sat. The dog barked, then ran! The bird sang; the` |
| 1 | natural | capital | with | low | 22 | 4 | ✅ | ✅ | `<p>The cat sat on the soft warm mat near the window watching` |
| 2 | natural | capital | without | high | 25 | 7 | ✅ | ✅ | `The cat sat. The dog barked, then ran! The bird sang; the fi` |
| 3 | natural | capital | without | low | 17 | 0 | ✅ | ✅ | `The cat sat on the soft warm mat near the window watching th` |
| 4 | natural | lowercase | with | high | 28 | 9 | ✅ | ✅ | `<p>the cat sat. the dog barked, then ran! the bird sang; the` |
| 5 | natural | lowercase | with | low | 22 | 4 | ✅ | ✅ | `<p>the cat sat on the soft warm mat near the window watching` |
| 6 | natural | lowercase | without | high | 25 | 7 | ✅ | ✅ | `the cat sat. the dog barked, then ran! the bird sang; the fi` |
| 7 | natural | lowercase | without | low | 17 | 0 | ✅ | ✅ | `the cat sat on the soft warm mat near the window watching th` |
| 8 | code | capital | with | high | 36 | 19 | ✅ | ✅ | `<code>If X: Y = Z + 1; A.run(); B, C = 2, 3; D[0] = E + F;</` |
| 9 | code | capital | with | low | 17 | 4 | ✅ | ✅ | `<code>If True Then Run Action Else Return Default Value Or E` |
| 10 | code | capital | without | high | 32 | 15 | ✅ | ✅ | `If X: Y = Z + 1; A.run(); B, C = 2, 3; D[0] = E + F; print Q` |
| 11 | code | capital | without | low | 12 | 0 | ✅ | ✅ | `If True Then Run Action Else Return Default Value Or End Blo` |
| 12 | code | lowercase | with | high | 36 | 19 | ✅ | ✅ | `<code>if x: y = z + 1; a.run(); b, c = 2, 3; d[0] = e + f;</` |
| 13 | code | lowercase | with | low | 17 | 4 | ✅ | ✅ | `<code>if true then run action else return default value or e` |
| 14 | code | lowercase | without | high | 32 | 15 | ✅ | ✅ | `if x: y = z + 1; a.run(); b, c = 2, 3; d[0] = e + f; print q` |
| 15 | code | lowercase | without | low | 12 | 0 | ✅ | ✅ | `if true then run action else return default value or end blo` |
| 16 | random | capital | with | high | 24 | 13 | ✅ | ✅ | `<Foo> Bar! Baz, Quux. <Tag/>; End: More? Less; <End/>` |
| 17 | random | capital | with | low | 23 | 4 | ✅ | ✅ | `<Foo>Bar Baz Quux Tag End More Less Words Here Now Then Soon` |
| 18 | random | capital | without | high | 24 | 11 | ✅ | ✅ | `Foo, Bar! Baz; Quux. Tag: End? More; Less! Done. Try; Now.` |
| 19 | random | capital | without | low | 18 | 0 | ✅ | ✅ | `Foo Bar Baz Quux Tag End More Less Words Here Now Then Soon ` |
| 20 | random | lowercase | with | high | 24 | 13 | ✅ | ✅ | `<foo> bar! baz, quux. <tag/>; end: more? less; <end/>` |
| 21 | random | lowercase | with | low | 22 | 4 | ✅ | ✅ | `<foo>bar baz quux tag end more less words here now then soon` |
| 22 | random | lowercase | without | high | 24 | 11 | ✅ | ✅ | `foo, bar! baz; quux. tag: end? more; less! done. try; now.` |
| 23 | random | lowercase | without | low | 18 | 0 | ✅ | ✅ | `foo bar baz quux tag end more less words here now then soon ` |