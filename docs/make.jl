using Documenter, FiniteDiff

makedocs(
    modules = [FiniteDiff],
)

deploydocs(
    deps = Deps.pip("pygments", "mkdocs", "python-markdown-math"),
    repo = "github.com/johnmyleswhite/FiniteDiff.jl.git",
)
