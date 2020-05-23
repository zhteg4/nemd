NEMD_MODULE=module
NEMD_SCRIPTS=scripts

export NEMD_SRC=$HOME/nemd
export PYTHONPATH=$NEMD_SRC/$NEMD_MODULE
export PATH=$NEMD_SRC/$NEMD_SCRIPTS:$PATH

alias cdgmm='cd $NEMD_SRC'
alias yp='python3 -m yapf -i'
alias cdgmm='cd ~/nemd'
alias sshme='sshpass -p **** ssh -X tzhang1@memtfe.crc.nd.edu'

[[ -r "/usr/local/etc/profile.d/bash_completion.sh" ]] && . "/usr/local/etc/profile.d/bash_completion.sh"

if mount | grep /scr/tzhang/nemd > /dev/null; then
  echo /scr/tzhang/nemd mounted
else
  echo ZTzt19881022ZTzt | sshfs -o password_stdin tzhang1@memtfe.crc.nd.edu:/afs/crc.nd.edu/user/t/tzhang1/research/nemd_pa/git/nemd /scr/tzhang/nemd
fi

