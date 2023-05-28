docker run -it --runtime=nvidia \
	--gpus all -p 8888:8888 \
	--entrypoint /shared/docker/entrypoint.sh \
	-v `pwd`:/shared:rw \
	-e HOST_UID="$(id -u)" \
	-e HOST_GID="$(id -g)" \
	deepml \
	/bin/bash
	# jupyter lab --ip 0.0.0.0 --port 8888 --no-browser 

