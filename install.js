module.exports = {
    run: [

        {
            method: "shell.run",
            params: {
                message: "git clone https://github.com/apple/ml-sharp app/ml-sharp"
            }
        },
        // 2. Install (Windows + NVIDIA)
        {
            when: "{{platform === 'win32' && gpu === 'nvidia'}}",
            method: "shell.run",
            params: {
                path: "app/ml-sharp",
                conda: {
                    path: "../env",
                    python: "3.11"
                },
                message: [
                    "pip install uv",
                    "uv pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128",
                    "uv pip install ..\\wheels\\gsplat-1.5.3-cp311-cp311-win_amd64.whl --no-deps",
                    "uv pip install -r requirements.txt",
                    "uv pip install gradio==6.2.0",
                    "uv pip install ."
                ]
            }
        },
        // 3. Install (Linux + NVIDIA)
        {
            when: "{{platform === 'linux'}}",
            method: "shell.run",
            params: {
                path: "app/ml-sharp",
                conda: {
                    path: "../env",
                    python: "3.11"
                },
                message: [
                    "pip install uv",
                    "uv pip install ../wheels/gsplat-1.5.3-cp311-cp311-linux_x86_64.whl --no-deps",
                    "uv pip install -r requirements.txt",
                    "uv pip install gradio==6.2.0",
                    "uv pip install ."
                ]
            }
        },
        // 4. Install (Mac ARM - darwin)
        {
            when: "{{platform === 'darwin'}}",
            method: "shell.run",
            params: {
                path: "app/ml-sharp",
                conda: {
                    path: "../env",
                    python: "3.11"
                },
                message: [
                    "pip install uv",
                    "uv pip install ../wheels/gsplat-1.5.3-cp311-cp311-macosx_11_0_arm64.whl --no-deps",
                    "uv pip install -r requirements.txt",
                    "uv pip install gradio==6.2.0",
                    "uv pip install ."
                ]
            }
        },
        // 5. Install (Fallback: Windows without NVIDIA or other unsupported platforms)
        {
            when: "{{(platform !== 'win32' && platform !== 'linux' && platform !== 'darwin') || (platform === 'win32' && gpu !== 'nvidia')}}",
            method: "shell.run",
            params: {
                path: "app/ml-sharp",
                conda: {
                    path: "../env",
                    python: "3.11"
                },
                message: [
                    "pip install uv",
                    "uv pip install -r requirements.txt",
                    "uv pip install gradio==6.2.0",
                    "uv pip install ."
                ]
            }
        },
        // 4. Notify
        {
            method: "notify",
            params: {
                html: "Installation completed! Click Start."
            }
        }
    ]
}