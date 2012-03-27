i=0
while read line; do
  echo $((`cat $1 | wc -l`-$i)) "left"
  wget $line -o wget.log -O data.html
  if [ -e data.html ]; then
    rm data.html
  fi
  i=$(($i+1))
done < $1

rm wget_out
