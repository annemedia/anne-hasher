# ANNE Hasher - Proof of Spacetime Hasher
ANNE Hasher is a GUI and command line Proof of Spacetime pre-mining/hashing tool compatible with ANNE, download the release executable for your system. 

## Features
- windows, linux, macOS
- direct and async I/O
- SIMD support: sse2, avx, avx2, avx512f
- GPU support

## Binary files

https://github.com/annemedia/anne-hasher/releases

## Running ANNE Hasher

To generate nonces for your ANNE Miner use the GUI or run it in a terminal/Command Prompt/PowerShell:

##### Linux/macOS terminal
```shell
./anne-hasher --help
```
##### Windows Command Prompt/PowerShell
```shell
.\anne-hasher --help
```

##### Recommended options
```
--n - number of nonces per file - 381500 ≈ 100GB
--id - your Neuron ID (NID) aka ANNE ID
--path - where do you want your nonces
--sna - count of auto-hashing of sequential files, each with --n nonces, starting after the last existing nonce found in the --path (for example --n 381500 with --sna 10 combination will create ~1TB (10x100GB) worth of nonces)
--cpu - how many CPU cores (note too high allocation may impact OS stability)
--gpu - how many GPU cores (note too high allocation may impact OS stability)
```
### Example CLI usage

##### Linux/macOS terminal
```shell
./anne-hasher --n 381500 --id 1234567890123456789 --path /home/user/annehashes --sna 10 --cpu 4 --gpu 5
```
##### Windows Command Prompt/PowerShell
```shell
.\anne-hasher --n 381500 --id 1234567890123456789 --path C:\Users\User\Documents\annehashes --sna 10 --cpu 4 --gpu 5
```

## Build from Sources

 - First you need to install a Rust stable toolchain, check https://www.rust-lang.org/tools/install.
 - Binaries are in **target/debug** or **target/release**

``` shell
# build release with GPU support, and GUI:
cargo build --release [--features=opencl,gui]

# build debug
cargo build [--features=opencl,gui]
```

## Forked from

ANNE Hasher is a significant upgrade and based on https://github.com/signum-network/signum-plotter


## Support
For any help or issues, feel free to reach out at https://t.me/anne_unplugged 

## Limitations
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
