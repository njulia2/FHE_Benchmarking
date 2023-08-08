SCRIPTPATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
docker build -t fhe_testing .
docker run -it -v ${SCRIPTPATH}:/app fhe_testing
