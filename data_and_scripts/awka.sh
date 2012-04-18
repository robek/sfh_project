awk '{ print "localhost/data?ts="$1"\&tch="$2"\&thr="$4"\&ssnr="$7"\&snoise="$8"\&osnr="$10"\&onoise="$11 } ' $1 > send_link.data
awk '{ print "localhost/tide?ts="$1"\&tide="$2 }' $2 > send_tide.data
