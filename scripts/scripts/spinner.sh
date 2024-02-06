#!/bin/bash

# Source : adapted from Jonas Eberle on this page -> https://unix.stackexchange.com/questions/225179/display-spinner-while-waiting-for-some-process-to-finish

# Shows a spinner while another command is running. Randomly picks one of 12 spinner styles.
# @args command to run (with any parameters) while showing a spinner. 
# E.g. <spinner sleep 10>
# E.g. <sleep 10 & spinner $1>

shutdown() {
  tput cnorm # reset cursor
}
trap shutdown EXIT

cursorBack() {
  echo -en "\033[$1D"
  # Mac compatible, but goes back to first column always. See comments
  #echo -en "\r"
}

spinner() {
  # make sure we use non-unicode character type locale 
  # (that way it works for any locale as long as the font supports the characters)
  local LC_CTYPE=C

  # Start the command in the background and capture its process ID
  "$@" &
  local pid=$! # Process Id of the previous running command

#   case $(($RANDOM % 12)) in
  case 1 in
  0)
    local spin='⠁⠂⠄⡀⢀⠠⠐⠈'
    local charwidth=3
    ;;
  1)
    local spin='-\|/'
    local charwidth=1
    ;;
  2)
    local spin="▁▂▃▄▅▆▇█▇▆▅▄▃▂▁"
    local charwidth=3
    ;;
  3)
    local spin="▉▊▋▌▍▎▏▎▍▌▋▊▉"
    local charwidth=3
    ;;
  4)
    local spin='←↖↑↗→↘↓↙'
    local charwidth=3
    ;;
  5)
    local spin='▖▘▝▗'
    local charwidth=3
    ;;
  6)
    local spin='┤┘┴└├┌┬┐'
    local charwidth=3
    ;;
  7)
    local spin='◢◣◤◥'
    local charwidth=3
    ;;
  8)
    local spin='◰◳◲◱'
    local charwidth=3
    ;;
  9)
    local spin='◴◷◶◵'
    local charwidth=3
    ;;
  10)
    local spin='◐◓◑◒'
    local charwidth=3
    ;;
  11)
    local spin='⣾⣽⣻⢿⡿⣟⣯⣷'
    local charwidth=3
    ;;
  esac

  local i=0
  tput civis # cursor invisible
  while kill -0 $pid 2>/dev/null; do
    local i=$(((i + $charwidth) % ${#spin}))
    printf "%s" "${spin:$i:$charwidth}"

    cursorBack 1
    sleep .1
  done
  tput cnorm
  wait $pid # capture exit code
  return $?
}

# to write the spinner on same line use -n
echo -n sleeping mode :
# spinner sleep 10

cmd=$"sleep 10"
spinner $cmd

