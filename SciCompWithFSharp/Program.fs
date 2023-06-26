open IsingModel.IsingModel2D
// SolveMetropolis 2 5_000_000
// SolveMetropolis 2.269 30_000_000

open MachineLearning.GradientDescent
let buildScaledTrainingSet n : TrainingSetRow[] =
    let min = 1
    let max = n
    let range = max - min |> float
    let mid = range / 2.
    let scale i =
        (float i - mid)/range

    [| min .. max|]
    |> Array.map scale
    |> Array.map (fun f -> TrainingSetRow.apply( [| f |], f))

let sample = InputVector.apply [| 3.14159265235 |]
let runLinearRegressionMethodsOnTrainingSet (alpha : LearningRate) (sampleSize : int) =
    buildScaledTrainingSet sampleSize
    |> RunAndPredictAllLinearRegressionTypes sample alpha

runLinearRegressionMethodsOnTrainingSet (LearningRate.apply 1.0e-02) 10000 |> ignore
