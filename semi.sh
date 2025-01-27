python -u semi_train.py --model GCN --dataset cora --layer 3 --hidden 512 --dropout 0.7 --lr 0.001 --wd 5e-4 --train --test
python -u semi_train.py --model GCN --dataset citeseer --layer 2 --hidden 512 --dropout 0.5 --lr 0.001 --wd 5e-4 --train --test
python -u semi_train.py --model GCN --dataset pubmed --layer 2 --hidden 256 --dropout 0.7 --lr 0.005 --wd 5e-4 --train --test

python -u semi_train.py --model GAT --dataset cora --layer 3 --hidden 512 --dropout 0.2 --lr 0.001 --wd 5e-4 --train --test
python -u semi_train.py --model GAT --dataset citeseer --layer 3 --hidden 256 --dropout 0.5 --lr 0.001 --wd 5e-4 --train --test
python -u semi_train.py --model GAT --dataset pubmed --layer 2 --hidden 512 --dropout 0.5 --lr 0.01 --wd 5e-4 --train --test

python -u semi_train.py --model APPNP --dataset cora --layer 8 --hidden 64 --alpha 0.1 --dropout 0.5 --lr 0.01 --wd 5e-4 --train --test
python -u semi_train.py --model APPNP --dataset citeseer --layer 8 --hidden 64 --alpha 0.1 --dropout 0.5 --lr 0.01 --wd 5e-4 --train --test
python -u semi_train.py --model APPNP --dataset pubmed --layer 8 --hidden 64 --alpha 0.1 --dropout 0.5 --lr 0.01 --wd 5e-4 --train --test

python -u semi_train.py --model GCNII --dataset cora --layer 64 --hidden 64 --alpha 0.1 --lamda 0.5 --dropout 0.6 --lr 0.01 --wd1 0.01 --wd2 5e-4 --train --test
python -u semi_train.py --model GCNII --dataset citeseer --layer 32 --hidden 256 --alpha 0.1 --lamda 0.6 --dropout 0.7 --lr 0.01 --wd1 0.01 --wd2 5e-4 --train --test
python -u semi_train.py --model GCNII --dataset pubmed --layer 16 --hidden 256 --alpha 0.1 --lamda 0.4 --dropout 0.5 --lr 0.01 --wd1 5e-4 --wd2 5e-4 --train --test

python -u semi_train.py --model GCNIII --dataset cora --layer 64 --hidden 64 --alpha 0.1 --lamda 0.5 --gamma 0.02 --dropout 0.6 --lr 0.01 --wd1 0.01 --wd2 5e-4 --intersect_memory --initial_residual --identity_mapping --train --test --seed 4
python -u semi_train.py --model GCNIII --dataset citeseer --layer 16 --hidden 256 --alpha 0.1 --lamda 0.6 --gamma 0.01 --dropout 0.5 --lr 0.01 --wd1 0.01 --wd2 5e-4 --intersect_memory --initial_residual --identity_mapping --train --test --seed 7
python -u semi_train.py --model GCNIII --dataset pubmed --layer 16 --hidden 256 --alpha 0.1 --lamda 0.4 --gamma 0.02 --dropout 0.5 --lr 0.01 --wd1 5e-4 --wd2 5e-4 --intersect_memory --initial_residual --identity_mapping --train --test --seed 3