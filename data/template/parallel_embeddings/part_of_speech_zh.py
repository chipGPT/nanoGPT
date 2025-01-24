import jieba.posseg as pseg

text = "他今天在北京大学的图书馆里看书，学习非常认真。这本书很有意思，内容包括历史、哲学和科学。"

words = pseg.cut(text)

for word, flag in words:
    print(f"{word}: {flag}")
