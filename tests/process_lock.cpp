#include "../src/process_lock.h"
#include <cstdio>
#include <sys/wait.h>
int main() {
    {
        eg2::ProcessLock owner;
        pid_t pid=fork();
        if (pid<0) return 1;
        if (!pid) {
            try { eg2::ProcessLock duplicate; _exit(1); }
            catch (const std::exception &e) {
                _exit(std::string(e.what()).find("another EmbeddingGemma")!=std::string::npos ? 0 : 1);
            }
        }
        int status=0;
        if (waitpid(pid,&status,0)!=pid || !WIFEXITED(status) || WEXITSTATUS(status)) return 1;
    }
    eg2::ProcessLock reacquire;
    std::puts("{\"test\":\"process-lock\",\"cases\":2,\"failures\":0}");
}
