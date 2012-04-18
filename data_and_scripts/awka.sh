HOST="localhost:8000"

awk '{ print "/data?ts="$1"\&tch="$2"\&thr="$4"\&ssnr="$7"\&snoise="$8"\&osnr="$10"\&onoise="$11 }' $1 > send_link.temp
awk '{ print "/tide?ts="$1"\&tide="$2 }' $2 > send_tide.temp

if [ -e send_link.data ]; then
  rm send_link.data
fi

if [ -e send_tide.data ]; then
  rm send_tide.data
fi


while read line; do
  echo $HOST$line >> send_link.data
done < send_link.temp

while read line; do
  echo $HOST$line >> send_tide.data
done < send_tide.temp

rm send_tide.temp send_link.temp
