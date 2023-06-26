namespace IsingModel

module IsingModel2D =

    [<Measure>] type Temperature
    [<Measure>] type Energy

    // This is the size of the 2D lattice.
    // This is a module-level parameter; wouldn't it be nice to have parametrized modules in F#?

    let private L = 15

    // This is the interaction matrix for the problem.
    // This is also a module-level parameter!
    //
    //    In this case we're modelling a ferromagnet with Jij > 0
    let private Jij =
        Array3D.init<float> L L 4 (fun _ _ _ -> 1.0)

    type private Direction =
        | N | E | W | S
    with
        member this.unapply =
            match this with
            | N -> 0
            | E -> 1
            | W -> 2
            | S -> 3
        static member All =
            [| N; E; W; S |]

    let private neighbour (x, y) direction =
        match direction with
        | E -> ((x + 1) % L), (y)
        | W -> ((x + L - 1) % L), (y)
        | N -> (x), ((y + 1) % L)
        | S -> (x), ((y + L - 1) % L)

    type Spin =
        | Up
        | Dn
    with
        static member apply = function
            | 0 -> Up
            | _ -> Dn

        static member random =
            System.Random.Shared.Next(2) |> Spin.apply

        static member (~-) (σ : Spin) =
            match σ with
            | Up -> Dn
            | Dn -> Up

        static member (*) (σi : Spin, σj : Spin) : float<Energy> =
            if (σi = σj ) then -1.0<Energy> else +1.0<Energy>

        override σ.ToString() =
            match σ with
            | Up -> "+ "
            | Dn -> "0 "

    type private Ising2D = { Spins : Spin[,]; Hamiltonian : float<Energy> }
    with
        static member apply (s) =
            let input = { Spins = s; Hamiltonian = 0.0<Energy> }
            { input with Hamiltonian = input.ComputeHamiltonian () }

        static member random () =
            (fun _ _ -> Spin.random)
            |> Array2D.init<Spin> L L
            |> Ising2D.apply

        member private this.ComputeSiteEnergy (flip: bool) (x, y) : float<Energy> =
            let spin' = this.Spins[x, y]
            let spin = if flip then (- spin') else spin'

            let interaction_with_direction (direction : Direction) =
                let (nx, ny) = neighbour(x, y) direction
                let neighbour_spin = this.Spins[nx, ny]
                Jij[x, y, direction.unapply] * (spin * neighbour_spin)

            Direction.All
            |> Seq.map interaction_with_direction
            |> Seq.reduce (+)

        member private this.ComputeHamiltonian () : float<Energy> =
            let mutable h = 0.0<Energy>
            for y in 0 .. (L - 1) do
                for x in 0 .. (L - 1) do
                    h <- h + this.ComputeSiteEnergy false (x, y)
            h

        member this.SiteEnergy        = this.ComputeSiteEnergy false
        member this.FlippedSiteEnergy = this.ComputeSiteEnergy true

        override this.ToString() =
            let spins = this.Spins
            let sb = System.Text.StringBuilder()
            ignore <| sb.AppendLine("Ising2D : \n{")
            ignore <| sb.AppendLine("spins : \n\t{")
            for y in 0 .. (L - 1) do
                ignore <| sb.Append("\t\t[ ")
                for x in 0 .. (L - 1) do
                    ignore <| sb.Append spins[x,y]
                ignore <| sb.AppendLine("]")
            ignore <| sb.AppendLine("\t}")
            ignore <| sb.AppendLine($"ham : {this.Hamiltonian}")
            ignore <| sb.AppendLine("}")

            sb.ToString()

    let SolveMetropolis (T : float<Temperature>) (num_iterations : int) =
        let mutable lattice = Ising2D.random()
        let mutable flips = 0

        let evolve (x, y) =
            let before = lattice.SiteEnergy        (x, y)
            let after  = lattice.FlippedSiteEnergy (x, y)
            let dE = after - before
            let accept =
                if dE <= 0.0<Energy>
                then true
                else
                    do flips <- flips + 1
                    let beta_delta_E = -dE/T // Energy/Temperature
                    let probability = exp(float beta_delta_E)
                    let random = System.Random.Shared.NextDouble()
                    probability >= random

            if accept then
                lattice.Spins[x, y] <- -lattice.Spins[x, y]
                lattice <- { lattice with Hamiltonian = lattice.Hamiltonian + dE }

        let start = System.DateTime.Now
        printfn "Before : \n%O : %s" lattice (start.ToString("r"))

        for _ in 0..num_iterations do
            evolve (System.Random.Shared.Next(L), System.Random.Shared.Next(L))

        let finish = System.DateTime.Now
        printfn "After : \n%O : %s" lattice (finish.ToString("r"))

        let duration = finish - start
        printfn $"F# - Solving Ising2D {L} x {L} ({num_iterations} iterations) at {T}K took {duration.TotalMilliseconds} ms. {flips} spins were flipped."