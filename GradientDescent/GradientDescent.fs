namespace MachineLearning
open MathNet.Numerics.LinearAlgebra.Double

module GradientDescent =
    // represents a vector of n features and an intercept
    type InputVector =
        | VI of DenseVector
    with
        member inline this.unapply = match this with | VI v -> v

        // add an intercept to an array of floats
        static member inline apply (xs : float[]) =
            [|
                yield  1.
                yield! xs
            |]
            |> (DenseVector >> VI)

        // add the intercept to a feature vector
        static member inline apply (v : DenseVector) =
            v.Values
            |> InputVector.apply

        // lift things convertible into floats to a feature vector
        static member inline apply<'a when ^a : (static member op_Explicit : ^a -> float)> (xs : ^a[]) =
            xs
            |> (Array.map float >> InputVector.apply)

        override this.ToString() =
            $"[ {this.unapply.ToRowMatrix().ToMatrixString().TrimEnd()} ]"

    type Label =
        | L of float
    with
        member inline this.unapply = match this with L l -> l

        static member inline apply<'a when ^a : (static member op_Explicit : ^a -> float)> (x : ^a) =
            (float >> L) x

        override this.ToString() =
            $"{this.unapply}"

    type TrainingSetRow =
        {
            x : InputVector
            y : Label
        }
    with
        static member inline apply<'a, 'b
            when ^a : (static member op_Explicit : ^a -> float)
             and ^b : (static member op_Explicit : ^b -> float)>
            (xs : ^a[], y : ^b) =
            {
                x = InputVector.apply xs
                y = Label.apply y
            }

    // represents a vector of dimension (n+1) of parameters
    type ParameterVector =
        | PV of DenseVector
    with
        member inline this.unapply = match this with | PV x -> x
        static member Allocate =
            Array.zeroCreate
            >> DenseVector.OfArray
            >> PV

        static member (<.>) (l : ParameterVector, r : InputVector) : float =
            l.unapply.DotProduct(r.unapply)

        override this.ToString() =
            $"[ {this.unapply.ToRowMatrix().ToMatrixString().TrimEnd()} ]"

    type LearningRate =
        | LR of float
    with
        member inline this.unapply = match this with | LR c -> c
        static member inline apply = LR

    type Error =
        | Err of float
    with
        member inline this.unapply = match this with | Err c -> c
        static member inline apply = Err

    // hypothesis function : R^(n+1) -> R^(n+1) -> R
    type HypothesisFunction = ParameterVector -> InputVector -> Label

    // gradient descent step function
    type GradientDescentStepFunction = ParameterVector -> ParameterVector

    // iteration, θ_prev, θ_curr
    type ConvergenceCheckFunction = int * ParameterVector * ParameterVector -> bool

    // GradientDescent is a sequence generator of steps
    let GradientDescent (seed : ParameterVector) (convergenceCheck : ConvergenceCheckFunction) (stepFunction : GradientDescentStepFunction) =
        let sequenceStep (i, theta) =
            let theta_new = stepFunction theta
            if convergenceCheck(i, theta, theta_new) then
                None
            else
                Some ((i + 1, theta_new), (i + 1, theta_new))
        in
        Seq.unfold sequenceStep (0, seed)

    let Predict (sample: InputVector) (theta : ParameterVector) : Label =
        sample.unapply.DotProduct(theta.unapply)
        |> Label.apply

    type RunType =
        | BatchGradientDescentLinearRegression of {| Epsilon : Error |}
        | MiniBatchStochasticGradientDescentLinearRegression of {| BatchSize : int; StepCount : int |}
        | StochasticGradientDescentLinearRegression of {| StepCount : int |}
        | NormalEquation

    // batch gradient descent for linear regression step function in vector form
    let lr_bgd_step (α : LearningRate) (rows : TrainingSetRow[]) : GradientDescentStepFunction =
        let m = float rows.Length
        fun (θt : ParameterVector) ->
            let gradients =
                rows
                |> Array.map (fun row ->
                    let x, y = row.x.unapply, row.y.unapply
                    ((θt <.> row.x) - y) * x)
                |> Array.reduce (+)

            θt.unapply - α.unapply/m * gradients
            |> PV

    // mini-batch stochastic gradient descent for linear regression step function in vector form
    let lr_mbgd_step (batchSize: int) (α : LearningRate) (rows : TrainingSetRow[]) : GradientDescentStepFunction =
        [| 0 .. batchSize |]
        |> Array.map (fun _ -> System.Random.Shared.Next(0, rows.Length))
        |> Array.map (fun i -> rows[i])
        |> lr_bgd_step α

    // stochastic gradient descent for linear regression step function in vector form
    let lr_sgd_step (α : LearningRate) (rows : TrainingSetRow[]) : GradientDescentStepFunction =
        lr_mbgd_step 1 α rows

    // normal equation
    let lr_normal (rows : TrainingSetRow[]) : ParameterVector =
        let X =
            rows
            |> Seq.map (fun row -> row.x.unapply.Values |> Seq.ofArray)
            |> Matrix.Build.DenseOfRows

        let y =
            rows
            |> Seq.map (fun row -> row.y.unapply)
            |> Vector.Build.DenseOfEnumerable

        let Xt = X.Transpose()

        ((Xt * X).Inverse()) * Xt * y
        |> DenseVector.OfVector
        |> PV

    // check for convergence by comparing with previous value
    let errorConvergenceCheck (epsilon : Error) : ConvergenceCheckFunction =
        fun (_ : int, θ_prev : ParameterVector, θ_curr: ParameterVector) ->
            let delta  = (θ_curr.unapply - θ_prev.unapply).L2Norm()
            if (System.Double.IsNaN delta)
            then
                failwith "Function has diverged"
            (delta <= epsilon.unapply)

    // check for convergence by number of iterations
    let countConvergenceCheck (max : int) : ConvergenceCheckFunction =
        fun (iteration : int, _ : ParameterVector, _: ParameterVector) ->
            iteration >= max

    let private timeAndRun (tag : string) f =
        let sw = System.Diagnostics.Stopwatch()
        async {
            sw.Restart()
            printfn $"Begin {tag}"
            let result = f ()
            sw.Stop()
            printfn $"End {tag} : {sw.ElapsedMilliseconds} ms\n"
            return result
        }

    let private processGradientDescentSteps (finalRowProcessor : int * ParameterVector -> 'a) (printStep : bool) (steps : (int * ParameterVector) seq) =
        let printStep (iterationId : int, theta : ParameterVector) =
            if printStep then
                printfn $"{iterationId}\t,\t[ {theta} ]"
            (iterationId, theta)

        steps
        |> Seq.map printStep
        |> Seq.maxBy fst
        |> finalRowProcessor

    let RunLinearRegression (finalRowProcessor : int * ParameterVector -> 'a) (printStep : bool) (alpha : LearningRate) (rows : TrainingSetRow[]) (runType: RunType) =
        let dimension = rows[0].x.unapply.Count
        let seed = ParameterVector.Allocate dimension

        match runType with
        | BatchGradientDescentLinearRegression args ->
            timeAndRun $"(M = {rows.Length}, d = {dimension}) Batch (epsilon = {args.Epsilon}) Gradient Descent" (fun () ->
                GradientDescent (seed) (errorConvergenceCheck args.Epsilon) (lr_bgd_step alpha rows)
                |> processGradientDescentSteps finalRowProcessor printStep)
        | StochasticGradientDescentLinearRegression args ->
            timeAndRun $"(M = {rows.Length}, d = {dimension}) Stochastic (iters: {args.StepCount}) Gradient Descent" (fun () ->
                GradientDescent (seed) (countConvergenceCheck args.StepCount) (lr_sgd_step alpha rows)
                |> processGradientDescentSteps finalRowProcessor printStep)
        | MiniBatchStochasticGradientDescentLinearRegression args ->
            timeAndRun $"(M = {rows.Length}, d = {dimension}) Mini-Batch (batch:{args.BatchSize}) Stochastic (iters: {args.StepCount}) Gradient Descent" (fun () ->
                GradientDescent (seed) (countConvergenceCheck args.StepCount) (lr_mbgd_step args.BatchSize alpha rows)
                |> processGradientDescentSteps finalRowProcessor printStep)
        | NormalEquation ->
            timeAndRun $"(M = {rows.Length}, d = {dimension}) Normal Equation" (fun () ->
                lr_normal rows
                |> (fun theta -> finalRowProcessor(0, theta)))

    let RunAndPredictAllLinearRegressionTypes (sample : InputVector) (alpha : LearningRate) (rows : TrainingSetRow[]) =
        let predictSample (_, theta) =
            let prediction = Predict sample theta
            printfn $"{{ Theta: {theta}; Sample: {sample}; Prediction: {prediction}; }}"
            theta

        [
            NormalEquation
            StochasticGradientDescentLinearRegression {| StepCount = min rows.Length 10_000 |}
            MiniBatchStochasticGradientDescentLinearRegression {| BatchSize = 100; StepCount = min rows.Length 10_000 |}
            BatchGradientDescentLinearRegression {| Epsilon = Error.apply 1.0e-10 |}
        ]
        |> Seq.map (RunLinearRegression predictSample false alpha rows)
        |> Async.Sequential
        |> Async.RunSynchronously
