start=`date +%s`
instanceNum=2
python3 correlation_test.py $instanceNum score | head -$instanceNum > score_1.txt &
python3 correlation_test.py $instanceNum score | head -$instanceNum > score_2.txt &
python3 correlation_test.py $instanceNum score | head -$instanceNum > score_3.txt &
python3 correlation_test.py $instanceNum score | head -$instanceNum > score_4.txt &

python3 correlation_test.py $instanceNum  edge | head -$instanceNum > edge_1.txt &
python3 correlation_test.py $instanceNum  edge | head -$instanceNum > edge_2.txt &
python3 correlation_test.py $instanceNum  edge | head -$instanceNum > edge_3.txt &
python3 correlation_test.py $instanceNum  edge | head -$instanceNum > edge_4.txt &

wait
cat score_1.txt score_2.txt score_3.txt score_4.txt | sort -n > all_scores.txt
cat edge_1.txt edge_2.txt edge_3.txt edge_4.txt | sort -n > all_edges.txt
end=`date +%s`

echo $((end-start))
