#! /bin/bash

function usage() {
    echo -e "\n Usage: $0 {old,new} <path>\n"
    echo "This script downloads the Sense2Vec archive to the given path and decompress it"
    echo "The first argument is the version of Sense2Vec to download"
    echo "where the version can be:"
    echo "    - 'new' (trained on Reddit 2019 archive)"
    echo "    - 'old' (trained on Reddit 2015 archive)"
    echo "The second argument is the output directory to download & decompress the archive to"
    echo -e "Example: bash $0 new data/s2v_new\n"
    exit 1
}

function download_file() {
    local url=$1
    local file=$2
    if [ ! -f $file ]; then
        wget -q -O $file $url
    fi
}

function main() {
    if [ $# -ne 2 ]; then
        usage
    fi
    local version="$1"
    local target_dir="$2"

    # check if target_dir is a directory
    if [ ! -d $target_dir ]; then
        echo "Creating $target_dir"
        mkdir -p $target_dir
    fi

    # check if target_dir is writable
    if [ ! -w $target_dir ]; then
        echo "$target_dir is not writable. Please check permissions"
        exit 1
    fi

    # download appropriate archive
    if [ $version == "old" ]; then
        local url="https://github.com/explosion/sense2vec/releases/download/v1.0.0/s2v_reddit_2015_md.tar.gz"
        basename="$(basename ${url})"
        archive_path="${target_dir}/${basename}"
        echo -n "Downloading $url to $target_dir ... "
        download_file $url ${target_dir}/${basename}
        echo "done"
        subdir="s2v_old"
    elif [ $version == "new" ]; then
        # download all parts and then merge
        local url_prefix="https://github.com/explosion/sense2vec/releases/download/v1.0.0/s2v_reddit_2019_lg.tar.gz"
        basename="$(basename ${url_prefix})"
        for i in {1..3}; do
            local url="${url_prefix}.00${i}"
            local file="${target_dir}/${basename}.00${i}"
            echo -n "Downloading $url to $file ... "
            download_file $url $file
            echo "done"
        done
        archive_path="${target_dir}/${basename}"
        echo -n "Merging files ... "
        cat ${archive_path}.* > "${archive_path}"
        echo "done"
        subdir="s2v_reddit_2019_lg"
    else
        usage
    fi

    # decompress archive
    output_dir="${target_dir}/${basename%%.*}"
    mkdir -p $output_dir
    echo -n "Decompressing archive to $output_dir ... "
    tar -xzf $archive_path -C $output_dir
    echo "done"

    echo -e "\n=====\n"
    echo "Downloaded and decompressed Sense2Vec archive to $output_dir"
    echo "You can use it in code like this:"
    echo ""
    echo "    from sense2vec import Sense2Vec"
    echo "    s2v = Sense2Vec().from_disk(\"${output_dir}/${subdir}\")"
    echo ""
}

main $@