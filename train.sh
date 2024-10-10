#CNWL experiments
for noiseratio in 0.2 0.8 0.4 0.6; do
    noiseperc=$(echo $noiseratio*100 / 1 | bc)
    for seed in 1 2 3; do
	#PLS-LSA
	python main.py --dataset miniimagenet --epochs 200 --batch-size 256 --net preresnet18 --lr 0.1 --seed ${seed} --exp-name CNWL_${noiseperc}_pls_lsa --noise-ratio ${noiseratio} --mixup --pretrained pretrained/simclr-cnwl-${noiseratio}.ckpt --alternate-ep linear loss --pls
	#PLS-LSA+
	python main.py --dataset miniimagenet --epochs 200 --batch-size 256 --net preresnet18 --lr 0.1 --seed ${seed} --exp-name CNWL_${noiseperc}_pls_lsaplus --noise-ratio ${noiseratio} --mixup --pretrained pretrained/simclr-cnwl-${noiseratio}.ckpt --alternate-ep linear loss --co-train
    done
done

#Webvision, l=0 + pls penalty is slightly better
for seed in 1 2 3; do
    python main.py --dataset webvision --epochs 130 --batch-size 64 --net inception --lr 0.02 --seed $seed --exp-name webvis_pls_lsa --mixup --alternate-ep linear loss --pls --pretrained pretrained/simclr-webvis.ckpt --pls-penalty --l 0 --pls
    python main.py --dataset webvision --epochs 130 --batch-size 64 --net inception --lr 0.02 --seed $seed --exp-name webvis_pls_lsaplus --mixup --alternate-ep linear loss --co-train --pretrained pretrained/simclr-webvis.ckpt --pls-penalty --l 0 
done

#Webly-fg experiments, datasets to be found at https://github.com/NUST-Machine-Intelligence-Laboratory/weblyFG-dataset
#Download Inet weights from https://pytorch.org/vision/0.12/_modules/torchvision/models/resnet.html#resnet50:  https://download.pytorch.org/models/resnet50-0676ba61.pth and place specify the weight path using the --pretrained argument
for seed in 1 2 3; do
    python main.py --net resnet50 --dataset web-bird --epochs 110 --batch-size 32 --lr 0.006 --seed $seed --exp-name pls_lsa_inet --mixup --conv 7 --pls --alternate-ep linear loss --pretrained pretrained/resnet50_imagenet.pth.pth
    python main.py --net resnet50 --dataset web-car --epochs 110 --batch-size 32 --lr 0.006 --seed $seed --exp-name pls_lsa_inet --mixup --conv 7 --pls --alternate-ep linear loss --pretrained pretrained/resnet50_imagenet.pth.pth
    python main.py --net resnet50 --dataset web-aircraft --epochs 110 --batch-size 32 --lr 0.006 --seed $seed --exp-name pls_lsa_inet --mixup --conv 7 --pls --alternate-ep linear loss --pretrained pretrained/resnet50_imagenet.pth.pth

    python main.py --net resnet50 --dataset web-bird --epochs 110 --batch-size 32 --lr 0.006 --seed $seed --exp-name pls_lsaplus_inet --mixup --conv 7 --co-train --alternate-ep linear loss --pretrained pretrained/resnet50_imagenet.pth.pth
    python main.py --net resnet50 --dataset web-car --epochs 110 --batch-size 32 --lr 0.006 --seed $seed --exp-name pls_lsaplus_inet --mixup --conv 7 --co-train --alternate-ep linear loss --pretrained pretrained/resnet50_imagenet.pth.pth 
    python main.py --net resnet50 --dataset web-aircraft --epochs 110 --batch-size 32 --lr 0.006 --seed $seed --exp-name pls_lsaplus_inet --mixup --conv 7 --co-train --alternate-ep linear loss --pretrained pretrained/resnet50_imagenet.pth.pth
done

#CLIP experiments.
for noiseratio in 0.2 0.8 0.4 0.6; do
    noiseperc=$(echo $noiseratio*100 / 1 | bc)
    for seed in 1 2 3; do
	#PLS-LSA
	python main.py --dataset miniimagenet --epochs 50 --batch-size 256 --net clip-ViT-B-32 --lr 1e-5 --seed $seed --exp-name CNWL_${noiseperc}_pls_lsa --noise-ratio ${noiseratio} --mixup --alternate-ep linear loss --pls
	#PLS-LSA+
	python main.py --dataset miniimagenet --epochs 50 --batch-size 256 --net clip-ViT-B-32 --lr 1e-5 --seed $seed --exp-name CNWL_${noiseperc}_pls_lsaplus --noise-ratio ${noiseratio} --mixup --alternate-ep linear loss --co-train
    done
done

#Webvision
for seed in 1 2 3; do
    python main.py --dataset webvision --epochs 50 --batch-size 64 --net clip-ViT-B-32 --lr 1e-5 --seed $seed --exp-name webvis_pls_lsa --mixup --alternate-ep linear loss --pls --pls-penalty --l 0
    python main.py --dataset webvision --epochs 50 --batch-size 64 --net clip-ViT-B-32 --lr 1e-5 --seed $seed --exp-name webvis_pls_lsaplus --mixup --alternate-ep linear loss --co-train --pls-penalty --l 0
done
