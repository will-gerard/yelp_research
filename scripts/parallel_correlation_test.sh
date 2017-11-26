python3 correlation_test.py score | head -250 > score_1.txt &
python3 correlation_test.py score | head -250 > score_2.txt &
python3 correlation_test.py score | head -250 > score_3.txt &
python3 correlation_test.py score | head -250 > score_4.txt &

python3 correlation_test.py edge | head -250 > edge_1.txt &
python3 correlation_test.py edge | head -250 > edge_2.txt &
python3 correlation_test.py edge | head -250 > edge_3.txt &
python3 correlation_test.py edge | head -250 > edge_4.txt &

wait
cat score_1.txt score_2.txt score_3.txt score_4.txt | sort -n > all_scores.txt
cat edge_1.txt edge_2.txt edge_3.txt edge_4.txt | sort -n > all_edges.txt
