#ifndef EG2_PROCESS_LOCK_H
#define EG2_PROCESS_LOCK_H
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <stdexcept>
#include <string>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>

namespace eg2 {
// Acquire BEFORE Context::create (which registers GPU queues/trap state).
// All project GPU tools share this per-user lock. It does not lock CPU servers
// or external GPU applications. Never unlink the inode: that breaks flock.
class ProcessLock {
    int fd_=-1;
public:
    ProcessLock() {
        std::string path="/tmp/embeddinggemma-kfd-"+std::to_string(getuid())+".lock";
        int fd=open(path.c_str(),O_CREAT|O_RDWR|O_CLOEXEC|O_NOFOLLOW,0600);
        if (fd<0) throw std::runtime_error("KFD process lock: "+std::string(std::strerror(errno)));
        struct stat st{};
        if (fstat(fd,&st) || !S_ISREG(st.st_mode) || st.st_uid!=getuid() || (st.st_mode&077)) {
            close(fd); throw std::runtime_error("KFD process lock has unexpected ownership or mode");
        }
        if (flock(fd,LOCK_EX|LOCK_NB)) {
            int error=errno; close(fd);
            if (error==EWOULDBLOCK) throw std::runtime_error("another EmbeddingGemma KFD process is active; run GPU tools sequentially");
            throw std::runtime_error("KFD process lock: "+std::string(std::strerror(error)));
        }
        fd_=fd;
    }
    ~ProcessLock() { if (fd_>=0) close(fd_); }
    ProcessLock(const ProcessLock &)=delete;
    ProcessLock &operator=(const ProcessLock &)=delete;
};
}
#endif
