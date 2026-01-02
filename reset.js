module.exports = {
    run: [{
        method: "fs.rm",
        params: {
            path: "app/env"
        }
    }, {
        method: "fs.rm",
        params: {
            path: "app/ml-sharp"
        }
    }]
}
