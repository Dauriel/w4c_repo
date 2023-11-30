#!/usr/bin/bash

declare -A region_year_checkpoints_2019=(
	["boxi_0015"]="/kaggle/input/w4c-smatunet-checkpoints/checkpoints/2019/r0015y2019-latest.ckpt"
	["boxi_0034"]="/kaggle/input/w4c-smatunet-checkpoints/checkpoints/2019/r0034y2019-latest.ckpt"
	["boxi_0076"]="/kaggle/input/w4c-smatunet-checkpoints/checkpoints/2019/r0076y2019-latest.ckpt"
	["roxi_0004"]="/kaggle/input/w4c-smatunet-checkpoints/checkpoints/2019/r0004y2019-latest.ckpt"
	["roxi_0005"]="/kaggle/input/w4c-smatunet-checkpoints/checkpoints/2019/r0005y2019-latest.ckpt"
	["roxi_0006"]="/kaggle/input/w4c-smatunet-checkpoints/checkpoints/2019/r0006y2019-latest.ckpt"
	["roxi_0007"]="/kaggle/input/w4c-smatunet-checkpoints/checkpoints/2019/r0007y2019-latest.ckpt"
)
declare -A region_year_checkpoints_2020=(
        ["boxi_0015"]="/kaggle/input/w4c-smatunet-checkpoints/checkpoints/2020/r0015y2020-latest.ckpt"
        ["boxi_0034"]="/kaggle/input/w4c-smatunet-checkpoints/checkpoints/2020/r0034y2020-latest.ckpt"
        ["boxi_0076"]="/kaggle/input/w4c-smatunet-checkpoints/checkpoints/2020/r0076y2020-latest.ckpt"
        ["roxi_0004"]="/kaggle/input/w4c-smatunet-checkpoints/checkpoints/2020/r0004y2020-latest.ckpt"
        ["roxi_0005"]="/kaggle/input/w4c-smatunet-checkpoints/checkpoints/2020/r0005y2020-latest.ckpt"
        ["roxi_0006"]="/kaggle/input/w4c-smatunet-checkpoints/checkpoints/2020/r0006y2020-latest.ckpt"
        ["roxi_0007"]="/kaggle/input/w4c-smatunet-checkpoints/checkpoints/2020/r0007y2020-latest.ckpt"
)

cdir=models/configurations/;
sdirDef=submission.core;
gpuDef=0;

sdir=$sdirDef;
gpu=$gpuDef;

cbase="$1"; shift;
cpt="$1"; shift;
if [ -n "$1" ]; then
    sdir="$1"; shift;
fi
if [ -n "$1" ]; then
    gpu="$1"; shift;
fi

out=$sdir.zip;


cat <<EOF
$0 configuration:

  configBase:     ${cbase:-MISSING}
    in folder $cdir
EOF
if [ -f "$cdir$cbase" ]; then
    ls -l "$cdir$cbase";
else
    echo "    Choices include:"
    ls -1 "$cdir"|grep 'pred.yaml$'|sed -e 's/^/      /';
fi

cat <<EOF
  checkpoint:     ${cpt:-MISSING}
  submissionDir:  $sdir
  gpuID:          $gpu
  output:         $out

EOF
[ -f "$cpt" ] && ls -l "$cpt";


if [ -z "$cpt" -o -n "$1" ]; then
    cat <<EOF
usage:    $0 {configBase} {checkpoint} [submissionDir] [gpuID]

	  The {configBase} file will be read from $cdir.
	  Predictions will be collected in folder submissionDir.
	  If not specified the default is: $sdirDef
	  For stage-1 support set submissionDir to: submission
	  The default gpuID is $gpuDef. Use a single GPU.

Examples: $0  config_baseline_stage2-pred.yaml \\
	      lightning_logs/YOURMODEL/checkpoints/YOURCHECK.ckpt

EOF
    exit
fi

if [ -d "$sdir" ]; then
    echo Deleting existing $sdir in 9 seconds unless you abort by Ctrl-C
    sleep 9 && rm -rf "$sdir/*" || exit;
else
    mkdir "$sdir";
    if [ -d "$sdir" ]; then
	echo "Created $sdir"
    else
	echo "Could not create $sdir - aborting.";
	exit
    fi
fi

for y in 2019 2020; do
    d=$sdir/$y;
    [ -d "$d" ] || mkdir "$d";
    declare -n checkpoints="region_year_checkpoints_$y"
    for r in "${!checkpoints[@]}"; do
        echo /=== $r $y for ${checkpoints[$r]}
        cin=$cdir$cbase;
        cnew=${cbase%.yaml}-$$-$r-$y.yaml;
        cout=$cdir$cnew;
        sed -e "s/%REGION%/$r/g" -e "s/%YEAR%/$y/" -e "s/%SDIR%/$sdir/" \
            <"$cin" >"$cout";
        checkpoint="${checkpoints[$r]}"
        python train.py --gpus $gpu --mode predict --config_path "$cnew" --model "SmatSimVP"\
               --checkpoint "$checkpoint"
        rm $cout;
    done
done

if pushd $sdir; then
    echo /=== output summary
    ls -l */*
    fl=`find -type f|grep -v '/[.]'|sed -e 's/ /%20/g'`;
    for f in $fl; do
	f="${f//%20/ }";
	echo Compressing $f ...
	gzip -9f "$f" &
    done
    wait;
    echo ...zip packing $sdir
    [ -s "../$out" ] && rm -f "../$out";
    zip -0mr ../$out * -x .\* \*/.\*
    popd
    rmdir $sdir;
    ls -l $out;
else
    echo "Cannot change to $sdir - aborting"
    exit
fi
echo \\=== done
