@echo off

set models=gcn-moe gin-spmoe
set pred_lens=3 6 9 12
set EPOCH=60

for %%m in (%models%) do (
    for %%l in (%pred_lens%) do (
        for %%f in (%folds%) do (
                python main.py --model %%m --pred_len %%l --fold %%f --epoch %EPOCH%
            )
        )
)
echo All experiments completed.
