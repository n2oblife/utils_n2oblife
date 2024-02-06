#!/usr/bin/env bash
clear

# to output the logs and errors 
# command > out.log 2> err.log or /dev/null

# Default values (CHANGE)
VARIABLE1="var1"
VARIABLE2="var2"


# handling of the inputs for the bash file(CHANGE)
help(){
    echo "Usage --> scripting template - it aims to show how to parse the input of a bash file easily
            [-v1 | --variable1] : explanation of the variable1, by default=$VARIABLE1
            [-v2 | --variable2] : explanation of the variable1, by default=$VARIABLE2
    "
    exit 2
}

# Options (CHANGE)
SHORT=v1:,v2:,h
LONG=variable1:,variable2:,help
OPTS=$(getopt -a -n script_to_launch --options $SHORT --longoptions $LONG -- "$@")

VALID_ARGUMENTS=$# # Returns the count of arguments that are in short or long options

if [ "$VALID_ARGUMENTS" -eq 0 ]; then
    echo "You have to enter at least on argument" 
    help
fi

eval set -- "$OPTS"

# parse the arguments(CHANGE)
while :
do
  case "$1" in
    -v1 | --variable1 )
      VARIABLE1="$2"
      shift 2
      ;;
    -v2 | --variable2 )
      VARIABLE2="$2"
      shift 2
      ;;
    -h | --help)
      help
      exit 2
      ;;
    --)
      shift;
      break 
      ;;
    *)
      echo "Unexpected option : $1"
      help
      exit 2
      ;;
  esac
done

#now it is possible to use the variables in scripts using $name_of_variable