python -u full_train.py --model GCNII --data cora --layer 64 --hidden 64 --alpha 0.2 --lamda 0.5 --dropout 0.5 --lr 0.01 --wd 1e-4
python -u full_train.py --model GCNII --data citeseer --layer 64 --hidden 64 --alpha 0.5 --lamda 0.5 --dropout 0.5 --lr 0.01 --wd 5e-6
python -u full_train.py --model GCNII --data pubmed --layer 64 --hidden 64 --alpha 0.1 --lamda 0.5 --dropout 0.5 --lr 0.01 --wd 5e-6
python -u full_train.py --model GCNII --data chameleon --layer 8 --hidden 64 --alpha 0.2 --lamda 1.5 --dropout 0.5 --lr 0.01 --wd 5e-4
python -u full_train.py --model GCNII --data cornell --layer 16 --hidden 64 --alpha 0.5 --lamda 1 --dropout 0.5 --lr 0.01 --wd 1e-3
python -u full_train.py --model GCNII --data texas --layer 32 --hidden 64 --alpha 0.5 --lamda 1.5 --dropout 0.5 --lr 0.01 --wd 1e-4
python -u full_train.py --model GCNII --data wisconsin --layer 16 --hidden 64 --alpha 0.5 --lamda 1 --dropout 0.5 --lr 0.01 --wd 5e-4

python -u full_train.py --model GCNIII --data cora --layer 8 --hidden 64 --alpha 0.2 --lamda 0 --gamma 0.02 --dropout 0.5 --lr 0.01 --wd 1e-4 --intersect_memory --initial_residual
python -u full_train.py --model GCNIII --data citeseer --layer 8 --hidden 128 --alpha 0.5 --lamda 1 --gamma 0.02 --dropout 0.5 --lr 0.01 --wd 5e-6 --intersect_memory --initial_residual
python -u full_train.py --model GCNIII --data pubmed --layer 32 --hidden 64 --alpha 0.1 --lamda 0.5 --gamma 0.02 --dropout 0.6 --lr 0.01 --wd 5e-6 --intersect_memory --initial_residual --identity_mapping
python -u full_train.py --model GCNIII --data chameleon --layer 2 --hidden 64 --alpha 0 --lamda 0 --gamma 0.05 --dropout 0 --lr 0.01 --wd 5e-4  --intersect_memory
python -u full_train.py --model GCNIII --data cornell --layer 2 --hidden 64 --alpha 0.8 --lamda 1 --gamma 0.02 --dropout 0.5 --lr 0.01 --wd 1e-3 --intersect_memory --initial_residual --identity_mapping
python -u full_train.py --model GCNIII --data texas --layer 2 --hidden 64 --alpha 0.5 --lamda 1.5 --gamma 0.05 --dropout 0.5 --lr 0.01 --wd 1e-4 --intersect_memory --initial_residual --identity_mapping
python -u full_train.py --model GCNIII --data wisconsin --layer 3 --hidden 64 --alpha 0.6 --lamda 1 --gamma 0.1 --dropout 0.8 --lr 0.01 --wd 5e-4 --intersect_memory --initial_residual --identity_mapping
