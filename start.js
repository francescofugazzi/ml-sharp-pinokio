module.exports = {
    daemon: true,
    run: [
        {
            method: "shell.run",
            params: {
                path: "app",
                conda: {
                    path: "env",
                },
                message: "python app.py",
                on: [{
                    event: "/http:\\/\\/[0-9.:]+/",
                    done: true
                }]
            }
        },
        {
            // This step is executed immediately after the event above is triggered
            method: "local.set",
            params: {
                url: "{{input.event[0]}}"
            }
        }
    ]
}