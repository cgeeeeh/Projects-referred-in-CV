rt = open("global_word_set.txt", 'r', encoding="UTF-8")
word_set_final = []
word = rt.readline()
count_word = 0
while word:
    word = word.strip()
    if word not in word_set_final:
        word_set_final.append(word)
    print("\r", "finished", count_word)
    count_word = count_word + 1
    word=rt.readline()
rt.close()
print("read done")
wt = open("global_word_set.txt", "w", encoding="UTF-8")
for word in word_set_final:
    wt.write(word + "\n")
wt.close()
