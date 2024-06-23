using Documenter, GraphicalLasso

makedocs(
    modules=[GraphicalLasso],
    sitename="GraphicalLasso.jl",
    format=Documenter.HTML(
        prettyurls=get(ENV, "CI", nothing) == "true"
    ),
    pages=[
        "Home" => [
            "index.md"
        ],
    ],
    modules=[GraphicalLasso]
)

deploydocs(
    repo="github.com/ivanuricardo/GraphicalLasso.jl.git",
)
