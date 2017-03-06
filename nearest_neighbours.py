from gensim.models import Doc2Vec
import random
import sys
def main(input_file, output_file):
    
    

    model = Doc2Vec.load_word2vec_format(input_file , binary=False)

    word_vocab = [w for w in model.vocab if "_*" not in w]

    K = 15

    with open(output_file, 'w') as f:

        p_ids = random.sample(xrange(0, 100000), K)
        for _id in p_ids:
            f.write(str(_id) + '====================================================\n')
            for i in model.most_similar("_*" + str(_id), topn = 100):
                if "_*" not  in i[0]:
                    f.write(i[0].encode('utf8') + ' ' + str(i[1]).encode('utf8') + '\n')

        for w in random.sample(word_vocab, K):
                f.write(w.encode('utf8')+'===================================================\n')
                for i in model.most_similar(w, topn = 100):
                    if "_*" not  in i[0]:
                        f.write(i[0].encode('utf8') + ' ' + str(i[1]).encode('utf8') + '\n')
if __name__ == "__main__":
   main(sys.argv[1], sys.argv[2])
