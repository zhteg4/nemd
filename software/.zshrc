
# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/Users/tzhang/opt/anaconda3/bin/conda' 'shell.zsh' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/Users/tzhang/opt/anaconda3/etc/profile.d/conda.sh" ]; then
        . "/Users/tzhang/opt/anaconda3/etc/profile.d/conda.sh"
    else
        export PATH="/Users/tzhang/opt/anaconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

# Load version control information
autoload -Uz vcs_info
precmd() { vcs_info }

# Format the vcs_info_msg_0_ variable
zstyle ':vcs_info:git:*' formats 'on %b'

# Set up the prompt (with git branch name)
setopt PROMPT_SUBST
PROMPT='%n in ${PWD/#$HOME/~} ${vcs_info_msg_0_} %# '

alias cdg='cd /Users/tzhang/git/nemd'
alias cdw='cd /Users/tzhang/scr/work'
alias ls='ls --color'
