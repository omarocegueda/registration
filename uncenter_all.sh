mkdir -p uncentered
for name in *.txt; do
    echo "Uncentering: $name ...";
    cp "$name" "uncentered/$name"
done
