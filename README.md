# install
bash install.sh
cd env_service/environments/appworld && bash setup.sh

# run
conda activate agentevolver
cd /mmu_cd_ssd/zhangzhenyu06/workspace/test_agent
bash ./examples/run_basic.sh

conda activate appworld
python -m agentevolver.preprocess.main