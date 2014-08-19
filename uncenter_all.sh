mkdir -p uncentered
for name in *.txt; do
    echo "Uncentering: $name ...";
    python ./uncenter.py "$name" "uncentered/$name"
done
