import MeCab

# MeCabの初期化
mecab = MeCab.Tagger("-Owakati")

# 文章を形態素解析して単語に分割
text = "私は自然言語処理が好きです。"
result = mecab.parse(text)
print(result)
