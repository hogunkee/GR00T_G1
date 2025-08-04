#!/bin/bash
TAG=$1
./scripts/simul_eval_multivideo.sh 0 robocasa_gr1_full_fourier_hands/PnPCupToDrawerClose_GR1FullFourierHands_Env ${TAG} &&
./scripts/simul_eval_multivideo.sh 0 robocasa_gr1_full_fourier_hands/PnPPotatoToMicrowaveClose_GR1FullFourierHands_Env ${TAG} &&
./scripts/simul_eval_multivideo.sh 0 robocasa_gr1_full_fourier_hands/PnPMilkToMicrowaveClose_GR1FullFourierHands_Env ${TAG} &&
./scripts/simul_eval_multivideo.sh 0 robocasa_gr1_full_fourier_hands/PnPBottleToCabinetClose_GR1FullFourierHands_Env ${TAG} &&
./scripts/simul_eval_multivideo.sh 0 robocasa_gr1_full_fourier_hands/PnPWineToCabinetClose_GR1FullFourierHands_Env ${TAG} &&
./scripts/simul_eval_multivideo.sh 0 robocasa_gr1_full_fourier_hands/PnPCanToDrawerClose_GR1FullFourierHands_Env ${TAG} &&
./scripts/simul_eval_multivideo.sh 0 robocasa_gr1_full_fourier_hands/PosttrainPnPNovelFromCuttingboardToBasketSplitA_GR1FullFourierHands_Env ${TAG} &&
./scripts/simul_eval_multivideo.sh 0 robocasa_gr1_full_fourier_hands/PosttrainPnPNovelFromCuttingboardToCardboardboxSplitA_GR1FullFourierHands_Env ${TAG} &&
./scripts/simul_eval_multivideo.sh 0 robocasa_gr1_full_fourier_hands/PosttrainPnPNovelFromCuttingboardToPanSplitA_GR1FullFourierHands_Env ${TAG} &&
./scripts/simul_eval_multivideo.sh 0 robocasa_gr1_full_fourier_hands/PosttrainPnPNovelFromCuttingboardToPotSplitA_GR1FullFourierHands_Env ${TAG} &&
./scripts/simul_eval_multivideo.sh 0 robocasa_gr1_full_fourier_hands/PosttrainPnPNovelFromCuttingboardToTieredbasketSplitA_GR1FullFourierHands_Env ${TAG} &&
./scripts/simul_eval_multivideo.sh 0 robocasa_gr1_full_fourier_hands/PosttrainPnPNovelFromPlacematToBasketSplitA_GR1FullFourierHands_Env ${TAG} &&
./scripts/simul_eval_multivideo.sh 0 robocasa_gr1_full_fourier_hands/PosttrainPnPNovelFromPlacematToBowlSplitA_GR1FullFourierHands_Env ${TAG} &&
./scripts/simul_eval_multivideo.sh 0 robocasa_gr1_full_fourier_hands/PosttrainPnPNovelFromPlacematToPlateSplitA_GR1FullFourierHands_Env ${TAG} &&
./scripts/simul_eval_multivideo.sh 0 robocasa_gr1_full_fourier_hands/PosttrainPnPNovelFromPlacematToTieredshelfSplitA_GR1FullFourierHands_Env ${TAG} &&
./scripts/simul_eval_multivideo.sh 0 robocasa_gr1_full_fourier_hands/PosttrainPnPNovelFromPlateToBowlSplitA_GR1FullFourierHands_Env ${TAG} &&
./scripts/simul_eval_multivideo.sh 0 robocasa_gr1_full_fourier_hands/PosttrainPnPNovelFromPlateToCardboardboxSplitA_GR1FullFourierHands_Env ${TAG} &&
./scripts/simul_eval_multivideo.sh 0 robocasa_gr1_full_fourier_hands/PosttrainPnPNovelFromPlateToPanSplitA_GR1FullFourierHands_Env ${TAG} &&
./scripts/simul_eval_multivideo.sh 0 robocasa_gr1_full_fourier_hands/PosttrainPnPNovelFromPlateToPlateSplitA_GR1FullFourierHands_Env ${TAG} &&
./scripts/simul_eval_multivideo.sh 0 robocasa_gr1_full_fourier_hands/PosttrainPnPNovelFromTrayToCardboardboxSplitA_GR1FullFourierHands_Env ${TAG} &&
./scripts/simul_eval_multivideo.sh 0 robocasa_gr1_full_fourier_hands/PosttrainPnPNovelFromTrayToPlateSplitA_GR1FullFourierHands_Env ${TAG} &&
./scripts/simul_eval_multivideo.sh 0 robocasa_gr1_full_fourier_hands/PosttrainPnPNovelFromTrayToPotSplitA_GR1FullFourierHands_Env ${TAG} &&
./scripts/simul_eval_multivideo.sh 0 robocasa_gr1_full_fourier_hands/PosttrainPnPNovelFromTrayToTieredbasketSplitA_GR1FullFourierHands_Env ${TAG} &&
./scripts/simul_eval_multivideo.sh 0 robocasa_gr1_full_fourier_hands/PosttrainPnPNovelFromTrayToTieredshelfSplitA_GR1FullFourierHands_Env ${TAG}
