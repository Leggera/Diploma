#!/bin/bash
git clone https://github.com/mesnilgr/iclr15.git
cp iclr15/scripts/word2vec.c word2vec.c
gcc word2vec.c -o word2vec -lm -pthread -O3 -march=native -funroll-loops

function normalize_text {
  awk '{print tolower($0);}' < $1 | sed -e 's/\./ \. /g' -e 's/<br \/>/ /g' -e 's/"/ " /g' \
  -e 's/,/ , /g' -e 's/(/ ( /g' -e 's/)/ ) /g' -e 's/\!/ \! /g' -e 's/\?/ \? /g' \
  -e 's/\;/ \; /g' -e 's/\:/ \: /g' > $1-norm
}
if [ ! -d ./data ]
then
    wget http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
    tar -xvf aclImdb_v1.tar.gz

    for j in train/pos train/neg test/pos test/neg train/unsup; do
	for i in `ls aclImdb/$j`; do cat aclImdb/$j/$i >> temp; awk 'BEGIN{print;}' >> temp; done
	normalize_text temp
	mv temp-norm aclImdb/$j/norm.txt
	rm temp
    done

    mkdir data
    mv aclImdb/train/pos/norm.txt data/full-train-pos.txt
    mv aclImdb/train/neg/norm.txt data/full-train-neg.txt
    mv aclImdb/test/pos/norm.txt data/test-pos.txt
    mv aclImdb/test/neg/norm.txt data/test-neg.txt
    mv aclImdb/train/unsup/norm.txt data/train-unsup.txt
fi


cat ./data/full-train-pos.txt ./data/full-train-neg.txt ./data/test-pos.txt ./data/test-neg.txt ./data/train-unsup.txt > alldata.txt
awk 'BEGIN{a=0;}{print "_*" a " " $0; a++;}' < alldata.txt > alldata-id.txt
shuf alldata-id.txt > alldata-id-shuf.txt

sizes=('-size 75' '-size 300')
alphas=('-alpha 0.025' '-alpha 0.1')
windows=('-window 5' '-window 20')
negatives=('-negative 12' '-negative 50')
models=('-cbow 1 -sample 1e-5' '-cbow 1 -sample 1e-4' '-cbow 1 -sample 1e-3' '-cbow 0 -sample 1e-3' '-cbow 0 -sample 1e-2' '-cbow 0 -sample 1e-1')
default_parameters=('-size 150 -alpha 0.05 -window 10 -negative 25 -iter 25 -threads 4')
default_models=('-cbow 0 -sample 1e-2' '-cbow 1 -sample 1e-4')
mkdir time_w2v
time_fold="time_w2v/"
mkdir space_w2v
space_fold="space_w2v/"
for model in "${default_models[@]}"; do
    if [ "$model" == "-cbow 1 -sample 1e-4" ];
    then
	for size in "${sizes[@]}"; do
	    delete=("-size 150")
	    d_p=${default_parameters[@]/$delete}
	    #echo $d_p
	    #echo $size
	    w2v_out="word2vec_c ""$model""$size"".txt"
	    w2v_t="$time_fold""time_""$w2v_out"
	    (time (./word2vec -train alldata-id-shuf.txt -output "$space_fold""$w2v_out" -sentence-vectors 1 $size $model $d_p >> "$w2v_t")) &>> "$w2v_t"
	done
    fi
    if [ "$model" == "-cbow 1 -sample 1e-4" ];
    then
	for alpha in "${alphas[@]}"; do
	    delete=("-alpha 0.05")
	    d_p=${default_parameters[@]/$delete}
	    #echo $d_p
	    #echo $alpha
	    w2v_out="word2vec_c ""$model""$alpha"".txt"
	    w2v_t="$time_fold""time_""$w2v_out"
	    (time (./word2vec -train alldata-id-shuf.txt -output "$space_fold""$w2v_out" -sentence-vectors 1 $alpha $model $d_p >> "$w2v_t")) &>> "$w2v_t"
	done
    fi
    if [ "$model" == "-cbow 1 -sample 1e-4" ];
    then
	for window in "${windows[@]}"; do
	    delete=("-window 10")
	    d_p=${default_parameters[@]/$delete}
	    #echo $d_p
	    #echo $window
	    w2v_out="word2vec_c ""$model""$window"".txt"
	    w2v_t="$time_fold""time_""$w2v_out"
	    (time (./word2vec -train alldata-id-shuf.txt -output "$space_fold""$w2v_out" -sentence-vectors 1 $window $model $d_p >> "$w2v_t")) &>> "$w2v_t"
	done
    fi
    for negative in "${negatives[@]}"; do
	delete=("-negative 25")
	d_p=${default_parameters[@]/$delete}
	#echo $d_p
	#echo $negative
	w2v_out="word2vec_c ""$model""$negative"".txt"
	w2v_t="$time_fold""time_""$w2v_out"
	(time (./word2vec -train alldata-id-shuf.txt -output "$space_fold""$w2v_out" -sentence-vectors 1 $negative $model $d_p >> "$w2v_t")) &>> "$w2v_t"
    done
done
for model in "${models[@]}"; do
    d_p=${default_parameters[@]}
    w2v_out="word2vec_c ""$model"".txt"
    w2v_t="$time_fold""time_""$w2v_out"
    (time (./word2vec -train alldata-id-shuf.txt -output "$space_fold""$w2v_out" -sentence-vectors 1 $model $d_p >> "$w2v_t")) &>> "$w2v_t" 
done


